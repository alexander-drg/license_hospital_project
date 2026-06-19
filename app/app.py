from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from . import models, ranker, learner
import json
from pathlib import Path
from .ollama_ai import (
    analyze_cv_full,
    score_cv_for_job,
    extract_structured_fields,
    generate_interview_questions,
    is_online,
    parse_job_description,
    generate_batch_summary,
)
from .sql_candidate_parser import (
    list_candidates,
    list_needs,
    build_candidate_profile,
    build_need_profile,
    build_ollama_prompt_for_sql_match,
    list_database_tables,
    build_multi_table_schema_profile,
    build_table_ai_summary,
)
from .ollama_ai import analyze_sql_match_prompt, analyze_sql_tables


bp = Blueprint("main", __name__, template_folder="../web/templates")

DB = models.MemoryDB()
RANKER = None
LEARNER = None

DATA_DIR = Path("data")
CANDIDATES_FILE = DATA_DIR / "candidates_store.json"
JOBS_FILE = DATA_DIR / "jobs_store.json"
RESULTS_FILE = DATA_DIR / "results_store.json"


def save_empty_json_file(path, empty_value):
    DATA_DIR.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(empty_value, f, ensure_ascii=False, indent=2)



def rebuild_ranker():
    global RANKER, LEARNER
    if DB.jobs.empty:
        RANKER = None
        LEARNER = None
    else:
        RANKER = ranker.HybridRanker(DB)
        LEARNER = learner.SimpleLearner(RANKER)


rebuild_ranker()

def score_to_rating(score: float) -> dict:
    """
    Converts internal 0-1 compatibility score into a user-friendly 1-5 rating.
    This is only for display. The internal raw score is still used for ranking.
    """

    try:
        score = float(score)
    except Exception:
        score = 0.0

    if score >= 0.80:
        return {
            "stars": 5,
            "label": "Strong Match",
            "description": "Very high compatibility"
        }

    if score >= 0.65:
        return {
            "stars": 4,
            "label": "Good Match",
            "description": "Good compatibility"
        }

    if score >= 0.45:
        return {
            "stars": 3,
            "label": "Moderate Match",
            "description": "Partial compatibility"
        }

    if score >= 0.25:
        return {
            "stars": 2,
            "label": "Weak Match",
            "description": "Limited compatibility"
        }

    return {
        "stars": 1,
        "label": "Very Weak Match",
        "description": "Low compatibility"
    }

@bp.route("/")
def home():
    return render_template(
        "home.html",
        candidates=DB.candidates_list(),
        jobs=DB.jobs_list(),
        ollama_online=is_online()
    )


@bp.route("/candidate/<cid>")
def candidate(cid):
    cand = DB.get_candidate(cid)
    if not cand:
        return "Candidate not found", 404

    recs = []
    if RANKER is not None:
        recs = RANKER.recommend(cid, top_k=10, with_reasons=True)

    return render_template("candidate.html", candidate=cand, recs=recs, ollama_online=is_online())


@bp.route("/job/<jid>")
def job_detail(jid):
    job = DB.get_job(jid)
    if not job:
        return "Job not found", 404
    return render_template("job.html", job=job)


@bp.route("/upload-cvs", methods=["POST"])
def upload_cvs():
    from .cv import cv_to_candidate

    files = request.files.getlist("cvs")
    if not files:
        return jsonify({"error": "no files"}), 400

    for f in files:
        if not f or not f.filename:
            continue
        cand = cv_to_candidate(f)
        DB.add_candidate(cand)

    return redirect(url_for("main.home"))


@bp.route("/upload-jobs", methods=["POST"])
def upload_jobs():
    from .cv import read_any

    files = request.files.getlist("jobs")
    pasted_jobs = (request.form.get("job_texts") or "").strip()

    uploaded_count = 0
    failed_items = []

    for f in files:
        if not f or not f.filename:
            continue
        try:
            raw = read_any(f)
            if raw.strip():
                job = parse_job_description(raw)
                job["source_filename"] = f.filename or ""
                DB.add_job(job)
                uploaded_count += 1
        except Exception as e:
            failed_items.append(f.filename or "unknown file")

    if pasted_jobs:
        blocks = [b.strip() for b in pasted_jobs.split("\n---\n") if b.strip()]
        for idx, block in enumerate(blocks, start=1):
            try:
                job = parse_job_description(block)
                job["source_filename"] = f"Pasted job description #{idx}"
                DB.add_job(job)
                uploaded_count += 1
            except Exception:
                failed_items.append(f"pasted job #{idx}")

    rebuild_ranker()

    return redirect(url_for("main.home"))

@bp.route("/reset", methods=["POST"])
def reset():
    DB.clear_all()
    rebuild_ranker()
    return redirect(url_for("main.home"))



@bp.route("/results")
def results():
    if DB.candidates.empty or DB.jobs.empty:
        return render_template(
            "results.html",
            overall_summary="Upload at least one CV and one job description to run the analysis.",
            job_results=[],
            total_candidates=len(DB.candidates),
            total_jobs=len(DB.jobs),
        )

    if RANKER is None:
        rebuild_ranker()

    all_matches = []

    for _, crow in DB.candidates.iterrows():
        cand = crow.to_dict()

        # ---------------------------------------------------------
        # Domain gate: if the CV is not related to medical/healthcare,
        # do not run the expensive hybrid ranker.
        # Instead, attach a clear user-facing result for every job.
        # ---------------------------------------------------------
        if not cand.get("is_medical_related", True):
            for _, jrow in DB.jobs.iterrows():
                job = jrow.to_dict()

                all_matches.append({
                    "candidate_id": cand["id"],
                    "candidate_name": cand.get("name", ""),
                    "candidate_specialization": cand.get("specialization", ""),
                    "candidate_years_exp": cand.get("years_exp", ""),
                    "job_id": job["id"],
                    "job_title": job.get("title", ""),
                    "job_specialization": job.get("specialization", ""),

                    # internal score, used for sorting/debugging
                    "score": 0.0,

                    # user-friendly display score
                    "rating": 1,
                    "rating_label": "Different Profession",
                    "rating_description": "The candidate works in a different professional field and is not suitable for this medical role",

                    "short_reason": (
                        "The CV is mainly unrelated "
                        "and does not contain enough medical, clinical, radiology, or healthcare administration "
                        "experience to be considered relevant for this position."
                        
                        
                    ),
                })

            continue

        # ---------------------------------------------------------
        # Normal hybrid ranking path for medical/healthcare candidates
        # ---------------------------------------------------------
        recs = (
            RANKER.recommend(
                cand["id"],
                top_k=min(5, len(DB.jobs)),
                with_reasons=True
            )
            if RANKER
            else []
        )

        for r in recs:
            job = DB.get_job(r["job_id"])
            if not job:
                continue

            score = float(r.get("score", 0))
            rating = score_to_rating(score)

            all_matches.append({
                "candidate_id": cand["id"],
                "candidate_name": cand.get("name", ""),
                "candidate_specialization": cand.get("specialization", ""),
                "candidate_years_exp": cand.get("years_exp", ""),
                "job_id": job["id"],
                "job_title": job.get("title", ""),
                "job_specialization": job.get("specialization", ""),

                # internal score, used for sorting/debugging
                "score": round(score, 3),

                # user-friendly display score
                "rating": rating["stars"],
                "rating_label": rating["label"],
                "rating_description": rating["description"],

                "short_reason": (
                    "The system evaluated semantic alignment, skill overlap, experience relevance, "
                    "competency fit, and bilingual Romanian-English terminology normalization."
                ),
            })

    # Sort all candidate-job matches by internal score
    all_matches.sort(
     key=lambda x: (
        int(x.get("rating", 0) or 0),
        float(x.get("score", 0) or 0)
     ),
      reverse=True
    )
    # Group matches by job
    job_results = []
    for _, jrow in DB.jobs.iterrows():
        job = jrow.to_dict()

        top_for_job = [
            m for m in all_matches
            if m["job_id"] == job["id"]
        ][:3]

        job_results.append({
            "job": job,
            "matches": top_for_job
        })

    compact_candidates = DB.candidates[
        ["id", "name", "specialization", "years_exp", "skills", "summary"]
    ].fillna("").to_dict(orient="records")

    compact_jobs = DB.jobs[
        ["id", "title", "specialization", "city", "summary"]
    ].fillna("").to_dict(orient="records")

    try:
        overall_summary = generate_batch_summary(
            candidates=compact_candidates[:5],
            jobs=compact_jobs[:5],
            matches=all_matches[:5]
        )
    except Exception:
        overall_summary = (
            "Analysis completed. Top matches were generated successfully, "
            "but the final natural-language summary could not be produced in time."
        )

    DB.save_results(overall_summary, job_results)

    return render_template(
        "results.html",
        overall_summary=overall_summary,
        job_results=job_results,
        total_candidates=len(DB.candidates),
        total_jobs=len(DB.jobs),
    )

@bp.route("/results/saved")
def saved_results():
    return render_template(
        "results.html",
        overall_summary=DB.results.get("overall_summary", ""),
        job_results=DB.results.get("job_results", []),
        total_candidates=len(DB.candidates),
        total_jobs=len(DB.jobs),
    )


@bp.route("/api/recommendations")
def api_recommendations():
    cid = request.args.get("candidate_id")
    if not cid or RANKER is None:
        return jsonify([])
    return jsonify(RANKER.recommend(cid, top_k=10, with_reasons=True))


@bp.route("/api/feedback", methods=["POST"])
def api_feedback():
    payload = request.get_json(force=True)
    DB.log_interaction(payload)
    if LEARNER is not None:
        LEARNER.update_from_feedback(payload)
    return jsonify({"ok": True})


@bp.route("/api/ai/analyze/<cid>")
def ai_analyze(cid):
    cand = DB.get_candidate(cid)
    if not cand:
        return jsonify({"error": "candidate not found"}), 404
    result = analyze_cv_full(cand.get("profile_text", ""))
    return jsonify(result)


@bp.route("/api/ai/extract/<cid>")
def ai_extract(cid):
    cand = DB.get_candidate(cid)
    if not cand:
        return jsonify({"error": "candidate not found"}), 404
    result = extract_structured_fields(cand.get("profile_text", ""))
    return jsonify(result)


@bp.route("/api/ai/fit")
def ai_fit():
    cid = request.args.get("candidate_id")
    jid = request.args.get("job_id")
    if not cid or not jid:
        return jsonify({"error": "candidate_id and job_id required"}), 400

    cand = DB.get_candidate(cid)
    job = DB.get_job(jid)
    if not cand or not job:
        return jsonify({"error": "not found"}), 404

    result = score_cv_for_job(
        cand.get("profile_text", ""),
        job.get("description_text", ""),
        job.get("title", "")
    )
    return jsonify(result)


@bp.route("/api/ai/interview-questions/<cid>")
def ai_interview_questions(cid):
    cand = DB.get_candidate(cid)
    if not cand:
        return jsonify({"error": "candidate not found"}), 404
    job_title = request.args.get("job_title", "")
    questions = generate_interview_questions(cand.get("profile_text", ""), job_title)
    return jsonify({"questions": questions})


@bp.route("/api/ai/status")
def ai_status():
    return jsonify({"online": is_online(), "model": "llama3.2:3b"})

@bp.route("/reset-cvs", methods=["POST"])
def reset_cvs():
    DB.candidates = DB.candidates.iloc[0:0]

    save_empty_json_file(CANDIDATES_FILE, [])
    save_empty_json_file(RESULTS_FILE, {
        "overall_summary": "",
        "job_results": []
    })

    DB.results = {
        "overall_summary": "",
        "job_results": []
    }

    return redirect(url_for("main.home"))


@bp.route("/reset-jobs", methods=["POST"])
def reset_jobs():
    DB.jobs = DB.jobs.iloc[0:0]

    save_empty_json_file(JOBS_FILE, [])
    save_empty_json_file(RESULTS_FILE, {
        "overall_summary": "",
        "job_results": []
    })

    DB.results = {
        "overall_summary": "",
        "job_results": []
    }

    rebuild_ranker()

    return redirect(url_for("main.home"))

@bp.route("/sql-analyzer", methods=["GET"])
def sql_analyzer():
    try:
        tables = list_database_tables()
        error = ""
    except Exception as e:
        tables = []
        error = str(e)

    return render_template(
        "sql_analyzer.html",
        tables=tables,
        table_result=None,
        error=error
    )


@bp.route("/sql-analyzer/analyze", methods=["POST"])
def sql_analyzer_analyze():
    candidate_id = int(request.form.get("candidate_id"))
    need_id = int(request.form.get("need_id"))

    try:
        prompt_data = build_ollama_prompt_for_sql_match(candidate_id, need_id)

        if not prompt_data.get("success"):
            return render_template(
                "sql_analyzer.html",
                candidates=list_candidates(limit=50),
                needs=list_needs(limit=50),
                result=None,
                error=prompt_data.get("message", "SQL analysis failed.")
            )

        ai_result = analyze_sql_match_prompt(prompt_data["prompt"])

        result = {
            "candidate_profile": prompt_data["candidate_profile"],
            "need_profile": prompt_data["need_profile"],
            "ai_result": ai_result,
        }

        return render_template(
            "sql_analyzer.html",
            candidates=list_candidates(limit=50),
            needs=list_needs(limit=50),
            result=result,
            error=""
        )

    except Exception as e:
        return render_template(
            "sql_analyzer.html",
            candidates=list_candidates(limit=50),
            needs=list_needs(limit=50),
            result=None,
            error=str(e)
        )
        
        
@bp.route("/sql-analyzer/analyze-tables", methods=["POST"])
def sql_analyzer_analyze_tables():
    selected_tables = request.form.getlist("tables")
    limit_per_table = int(request.form.get("limit_per_table", 3))

    try:
        tables = list_database_tables()

        if not selected_tables:
            return render_template(
                "sql_analyzer.html",
                tables=tables,
                table_result=None,
                error="Please select at least one table."
            )

        profile = build_table_ai_summary(selected_tables) 
        ai_result = analyze_sql_tables(profile["profile_text"])

        table_result = {
            "tables": selected_tables,
            "profile_text": profile["profile_text"],
            "ai_result": ai_result,
        }

        return render_template(
            "sql_analyzer.html",
            tables=tables,
            table_result=table_result,
            error=""
        )

    except Exception as e:
        return render_template(
            "sql_analyzer.html",
            tables=list_database_tables(),
            table_result=None,
            error=str(e)
        )
        
        