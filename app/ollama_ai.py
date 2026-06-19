import json
import re
import uuid
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

# Fast model for extraction / parsing / scoring
OLLAMA_FAST_MODEL = "llama3.2:3b"

# Same model for summaries for now; you can change later
OLLAMA_SUMMARY_MODEL = "llama3.2:3b"


def is_online() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("No JSON object found in Ollama response")


def _ollama_json(prompt: str, timeout: int = 90) -> dict:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_FAST_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_predict": 180,
                "num_ctx": 2048,
                "num_thread": 8
            }
        },
        timeout=timeout
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    return json.loads(_extract_json_object(raw))


def _ollama_text(prompt: str, timeout: int = 90) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_SUMMARY_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "num_predict": 220,
                "num_ctx": 2048,
                "num_thread": 8
            }
        },
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()

def _safe_list(value):
    if isinstance(value, list):
        return value
    return []


def _safe_str(value):
    if value is None:
        return ""
    return str(value).strip()


# -------------------------------------------------------------------
# CV ANALYSIS
# -------------------------------------------------------------------

def fallback_extract_structured_fields(cv_text: str) -> dict:
    text = cv_text or ""

    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)
    phone_match = re.search(r"(\+?\d[\d\s().-]{7,})", text)

    cities = [
        "Bucharest", "Cluj", "Iași", "Timisoara", "Constanța",
        "Brașov", "Sibiu", "Craiova", "Oradea", "Arad"
    ]
    city = ""
    for c in cities:
        if re.search(rf"\b{re.escape(c)}\b", text, re.I):
            city = c
            break

    education_hits = []
    for kw in ["Bachelor", "Master", "PhD", "University", "College", "Licenta", "Masterat"]:
        if kw.lower() in text.lower():
            education_hits.append(kw)

    cert_hits = []
    for kw in ["ACLS", "BLS", "ATLS", "PALS", "GMP", "Google Ads", "Meta Blueprint"]:
        if kw.lower() in text.lower():
            cert_hits.append(kw)

    skill_vocab = [
        "python", "sql", "excel", "google ads", "meta ads", "tiktok", "linkedin",
        "ecg", "echocardiography", "primary care", "vaccination", "diagnostics",
        "telemedicine", "emr", "ehr", "communication", "sales", "marketing"
    ]
    skills = [s for s in skill_vocab if s in text.lower()]

    years = 0
    m = re.search(r"(\d{1,2})\s*(?:years?|ani)\s+(?:experience|exp)", text, re.I)
    if m:
        years = int(m.group(1))

    return {
        "education": education_hits,
        "employment_history": [],
        "languages": [],
        "certifications": cert_hits,
        "skills": skills,
        "years_experience": years,
        "phone": phone_match.group(1).strip() if phone_match else "",
        "city": city,
    }


def extract_structured_fields(cv_text: str) -> dict:
    compact_text = (cv_text or "").strip()[:5000]

    prompt = f"""
Extract structured candidate information from this resume text.

Return ONLY valid JSON.
No markdown. No explanations.

Schema:
{{
  "education": [],
  "employment_history": [],
  "languages": [],
  "certifications": [],
  "skills": [],
  "years_experience": 0,
  "phone": "",
  "city": ""
}}

Resume text:
----------------
{compact_text}
----------------
"""
    try:
        data = _ollama_json(prompt, timeout=75)
        return {
            "education": _safe_list(data.get("education")),
            "employment_history": _safe_list(data.get("employment_history")),
            "languages": _safe_list(data.get("languages")),
            "certifications": _safe_list(data.get("certifications")),
            "skills": _safe_list(data.get("skills")),
            "years_experience": int(data.get("years_experience", 0) or 0),
            "phone": _safe_str(data.get("phone")),
            "city": _safe_str(data.get("city")),
        }
    except Exception:
        return fallback_extract_structured_fields(compact_text)


def analyze_cv_full(cv_text: str) -> dict:
    compact_text = (cv_text or "").strip()[:2500]

    prompt = f"""
Analyze this CV and extract structured data.

Return ONLY valid JSON.

Schema:
{{
  "education": [],
  "employment_history": [],
  "languages": [],
  "certifications": [],
  "skills": [],
  "years_experience": 0,
  "phone": "",
  "city": "",
  "summary": "",
  "strengths": [],
  "risks": [],
  "seniority_guess": "",
  "recommended_roles": []
}}

Rules:
- Be concise.
- Do not invent information.
- Use empty values if missing.

CV text:
----------------
{compact_text}
----------------
"""
    try:
        data = _ollama_json(prompt, timeout=90)
        return data
    except Exception:
        fallback = fallback_extract_structured_fields(compact_text)
        fallback.update({
            "summary": compact_text[:300],
            "strengths": [],
            "risks": [],
            "seniority_guess": "",
            "recommended_roles": [],
        })
        return fallback

def generate_interview_questions(cv_text: str, job_title: str = "") -> list:
    compact_text = (cv_text or "").strip()[:4000]

    prompt = f"""
Generate 5 interview questions for this candidate.
Role context: {job_title}

Return ONLY valid JSON.

Schema:
{{
  "questions": []
}}

CV text:
----------------
{compact_text}
----------------
"""
    try:
        data = _ollama_json(prompt, timeout=60)
        return _safe_list(data.get("questions"))[:5]
    except Exception:
        return [
            "Can you walk me through your most relevant recent experience?",
            "What tools or systems have you used most often in your work?",
            "What type of role are you targeting next?",
            "What was your biggest challenge in a recent position?",
            "Why do you think you are a fit for this role?",
        ]


# -------------------------------------------------------------------
# CV vs JOB FIT
# -------------------------------------------------------------------

def score_cv_for_job(cv_text: str, job_text: str, job_title: str = "") -> dict:
    compact_cv = (cv_text or "").strip()[:3500]
    compact_job = (job_text or "").strip()[:3500]

    prompt = f"""
Evaluate how well the candidate CV matches the job.

Return ONLY valid JSON.

Schema:
{{
  "score": 0.0,
  "fit_label": "",
  "summary": "",
  "strengths": [],
  "gaps": []
}}

Job title: {job_title}

CV:
----------------
{compact_cv}
----------------

Job description:
----------------
{compact_job}
----------------
"""
    try:
        data = _ollama_json(prompt, timeout=75)

        score = data.get("score", 0)
        try:
            score = float(score)
        except Exception:
            score = 0.0

        return {
            "score": max(0.0, min(score, 1.0)),
            "fit_label": _safe_str(data.get("fit_label")),
            "summary": _safe_str(data.get("summary")),
            "strengths": _safe_list(data.get("strengths")),
            "gaps": _safe_list(data.get("gaps")),
        }
    except Exception:
        return {
            "score": 0.0,
            "fit_label": "Unknown",
            "summary": "Fit analysis could not be generated in time.",
            "strengths": [],
            "gaps": [],
        }


# -------------------------------------------------------------------
# JOB PARSING
# -------------------------------------------------------------------

def fallback_parse_job_description(job_text: str) -> dict:
    text = (job_text or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    title = lines[0][:120] if lines else "Untitled Job"

    city = ""
    for candidate_city in ["Bucharest", "Cluj", "Iași", "Timisoara", "Constanța", "Brașov", "Sibiu", "Craiova"]:
        if candidate_city.lower() in text.lower():
            city = candidate_city
            break

    lowered = text.lower()

    skill_vocab = [
        "python", "sql", "excel", "google ads", "meta ads", "tiktok", "linkedin",
        "primary care", "vaccination", "diagnostics", "ecg", "echocardiography",
        "telemedicine", "ehr", "emr", "patient care", "communication", "sales"
    ]
    found_skills = [s for s in skill_vocab if s in lowered]

    seniority = ""
    if "senior" in lowered:
        seniority = "Senior"
    elif "mid" in lowered or "middle" in lowered:
        seniority = "Mid"
    elif "junior" in lowered or "entry" in lowered:
        seniority = "Junior"

    contract_type = "Full-time" if "full-time" in lowered or "full time" in lowered else ""
    shift = "day" if "day" in lowered else ""

    return {
        "id": "j_" + uuid.uuid4().hex[:8],
        "title": title or "Untitled Job",
        "specialization": "",
        "seniority": seniority,
        "must_have_skills": ", ".join(found_skills[:8]),
        "nice_to_have_skills": "",
        "city": city,
        "salary_min": "",
        "salary_max": "",
        "contract_type": contract_type,
        "shift": shift,
        "description_text": text[:6000],
        "summary": text[:300],
        "source_filename": "",
    }


def parse_job_description(job_text: str) -> dict:
    compact_text = (job_text or "").strip()[:5000]

    prompt = f"""
You extract structured job data from a job description.

Return ONLY valid JSON.
No markdown. No explanations.

The "summary" field must:
- be 1 to 2 concise sentences
- describe the role in natural language
- mention core responsibilities and important required skills
- NOT copy phrases verbatim from the source
- NOT start with "Titlu:" or repeat raw formatting labels
- be written as a clean recruiter-style summary

Schema:
{{
  "title": "",
  "specialization": "",
  "seniority": "",
  "must_have_skills": [],
  "nice_to_have_skills": [],
  "city": "",
  "salary_min": "",
  "salary_max": "",
  "contract_type": "",
  "shift": "",
  "summary": ""
}}

Job description:
----------------
{compact_text}
----------------
"""
    try:
        data = _ollama_json(prompt, timeout=90)

        return {
            "id": "j_" + uuid.uuid4().hex[:8],
            "title": _safe_str(data.get("title")) or "Untitled Job",
            "specialization": _safe_str(data.get("specialization")),
            "seniority": _safe_str(data.get("seniority")),
            "must_have_skills": ", ".join(_safe_list(data.get("must_have_skills"))),
            "nice_to_have_skills": ", ".join(_safe_list(data.get("nice_to_have_skills"))),
            "city": _safe_str(data.get("city")),
            "salary_min": _safe_str(data.get("salary_min")),
            "salary_max": _safe_str(data.get("salary_max")),
            "contract_type": _safe_str(data.get("contract_type")),
            "shift": _safe_str(data.get("shift")),
            "description_text": compact_text,
            "summary": _safe_str(data.get("summary")),
            "source_filename": "",
        }
    except Exception:
        return fallback_parse_job_description(compact_text)


# -------------------------------------------------------------------
# FINAL BATCH SUMMARY
# -------------------------------------------------------------------
def build_deterministic_batch_summary(matches: list) -> str:
    if not matches:
        return (
            "The analysis was completed successfully, but no relevant candidate-job matches were found. "
            "This may happen when the uploaded CVs do not contain enough relevant experience for the available job descriptions."
        )

    def is_different_profession(match):
        label = str(match.get("rating_label", "")).lower()
        specialization = str(match.get("candidate_specialization", "")).lower()

        return (
            "different profession" in label
            or "unrelated" in label
            or "non-medical" in specialization
            or "different profession" in specialization
        )

    # Never allow unrelated/different profession candidates to be selected as strongest
    valid_matches = [
        m for m in matches
        if not is_different_profession(m)
    ]

    if valid_matches:
        best = sorted(
            valid_matches,
            key=lambda m: (
                int(m.get("rating", 0) or 0),
                float(m.get("score", 0) or 0)
            ),
            reverse=True
        )[0]
    else:
        best = sorted(
            matches,
            key=lambda m: (
                int(m.get("rating", 0) or 0),
                float(m.get("score", 0) or 0)
            ),
            reverse=True
        )[0]

    candidate = best.get("candidate_name", "The strongest candidate")
    job = best.get("job_title", "the selected position")
    rating = best.get("rating", "")
    rating_label = best.get("rating_label", "")
    rating_description = best.get("rating_description", "")
    reason = best.get("short_reason", "")

    different_profession_matches = [
        m for m in matches
        if is_different_profession(m)
    ]

    summary = (
        f"The analysis was completed successfully. Based on the generated ranking, the strongest candidate for "
        f"the {job} position is {candidate}, with a {rating}/5 rating classified as {rating_label}. "
    )

    if rating_description:
        summary += f"{rating_description}. "

    summary += (
        "The system evaluated the candidates using semantic alignment, skill overlap, experience relevance, "
        "competency fit, and bilingual Romanian-English terminology normalization where applicable. "
    )

    if reason:
        summary += f"{reason} "

    if different_profession_matches:
        names = ", ".join(
            sorted(set(
                m.get("candidate_name", "Another candidate")
                for m in different_profession_matches
            ))
        )

        summary += (
            f"The system also identified {names} as belonging to a different professional field or as insufficiently "
            f"aligned with the medical recruitment context. These candidates should not be considered suitable for the role "
            f"unless additional relevant healthcare experience is confirmed. "
        )

    summary += (
        "The result should be interpreted as a decision-support recommendation, not as an automatic hiring decision. "
        "A recruiter should still validate the final shortlist manually, especially when the role requires specific medical "
        "or clinical qualifications."
    )

    return summary


def generate_batch_summary(candidates: list, jobs: list, matches: list) -> str:
    """
    Controlled AI summary.

    The code decides the ranking facts.
    Ollama only writes the final explanation.
    """

    if not matches:
        return build_deterministic_batch_summary(matches)

    def is_different_profession(match):
        label = str(match.get("rating_label", "")).lower()
        specialization = str(match.get("candidate_specialization", "")).lower()
        reason = str(match.get("short_reason", "")).lower()

        return (
            "different profession" in label
            or "unrelated" in label
            or "non-medical" in specialization
            or "different profession" in specialization
            or "sales" in reason
            or "marketing" in reason
        )

    sorted_matches = sorted(
        matches,
        key=lambda m: (
            int(m.get("rating", 0) or 0),
            float(m.get("score", 0) or 0)
        ),
        reverse=True
    )

    valid_matches = [
        m for m in sorted_matches
        if not is_different_profession(m)
    ]

    different_profession_matches = [
        m for m in sorted_matches
        if is_different_profession(m)
    ]

    best = valid_matches[0] if valid_matches else sorted_matches[0]

    controlled_facts = {
        "strongest_candidate": {
            "candidate_name": best.get("candidate_name", ""),
            "candidate_specialization": best.get("candidate_specialization", ""),
            "job_title": best.get("job_title", ""),
            "rating": best.get("rating", ""),
            "rating_label": best.get("rating_label", ""),
            "rating_description": best.get("rating_description", ""),
            "reason": best.get("short_reason", "")
        },
        "other_valid_candidates": [
            {
                "candidate_name": m.get("candidate_name", ""),
                "candidate_specialization": m.get("candidate_specialization", ""),
                "job_title": m.get("job_title", ""),
                "rating": m.get("rating", ""),
                "rating_label": m.get("rating_label", ""),
                "reason": m.get("short_reason", "")
            }
            for m in valid_matches[1:4]
        ],
        "different_profession_candidates": [
            {
                "candidate_name": m.get("candidate_name", ""),
                "candidate_specialization": m.get("candidate_specialization", ""),
                "job_title": m.get("job_title", ""),
                "rating": m.get("rating", ""),
                "rating_label": m.get("rating_label", ""),
                "reason": m.get("short_reason", "")
            }
            for m in different_profession_matches[:4]
        ]
    }

    fallback = build_deterministic_batch_summary(sorted_matches[:10])

    prompt = f"""
Write a specific HR screening executive summary for a recruitment dashboard.

Use ONLY these controlled facts:
{json.dumps(controlled_facts, ensure_ascii=False)}

Important rules:
- Do not choose a different strongest candidate.
- The strongest candidate is already defined in controlled_facts.
- Do not say a Different Profession candidate is the best candidate.
- Do not invent medical experience.
- Do not invent job titles.
- Do not mention raw internal scores such as 0.174 or 0.221.
- Use the 1 to 5 rating scale.
- Mention if a candidate is from a different profession.
- Explain the result in a practical recruiter-friendly way.
- Maximum 7 sentences.
- Plain text only.
- Do not use markdown.
- Do not start with "Here is".
"""

    try:
        result = _ollama_text(prompt, timeout=90)

        if result and len(result.strip()) > 30:
            cleaned = result.strip()
            cleaned = cleaned.replace("Here is a specific HR screening executive summary:", "").strip()
            cleaned = cleaned.replace("Here is the HR screening executive summary:", "").strip()
            cleaned = cleaned.replace("Here is", "").strip()

            # Safety check: if Ollama somehow names a different-profession candidate as strongest, use fallback.
            first_sentence = cleaned.split(".")[0].lower()

            for m in different_profession_matches:
                bad_name = str(m.get("candidate_name", "")).lower()
                if bad_name and bad_name in first_sentence:
                    return fallback

            return cleaned

        return fallback

    except Exception as e:
        print("Ollama controlled summary failed:", repr(e))
        return fallback


def analyze_sql_match_prompt(prompt: str) -> str:
    short_prompt = prompt[:5000]

    try:
        return _ollama_text(short_prompt, timeout=60)
    except Exception as e:
        print("SQL Ollama analysis failed:", repr(e))
        return (
            "The SQL profiles were extracted successfully, but the AI analysis could not be "
            "completed in time. The candidate and job data can still be reviewed manually."
        )




def clean_ai_markdown(text: str) -> str:
    if not text:
        return ""

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("###", "")
    text = text.replace("##", "")
    text = text.replace("#", "")
    text = text.replace("* ", "- ")

    return text.strip()


def analyze_sql_tables(table_profile_text: str) -> str:
    compact_text = (table_profile_text or "").strip()[:1800]

    prompt = f"""
You are analyzing SQL tables from a recruitment database.

Write a clean dashboard-style paragraph.

Based only on this compact schema summary, explain:
- what the selected tables store
- how they relate to candidates, jobs, or needs
- how this data can support candidate-job matching
- any visible limitations

Rules:
- Plain text only.
- Do not use markdown.
- Do not use bullet points.
- Do not use asterisks.
- Do not use headings.
- Do not mention every column.
- Do not invent data.
- Maximum 3 short paragraphs.

Schema summary:
{compact_text}
"""

    try:
        result = _ollama_text(prompt, timeout=90)
        return clean_ai_markdown(result)
    except Exception as e:
        print("SQL table analysis failed:", repr(e))
        return (
            "The selected SQL tables were loaded successfully. They contain structured recruitment data "
            "such as candidates, education, experience, needs, and relational linking fields. These tables can support "
            "candidate-job matching by combining profile information, work history, requirements, and related attributes. "
            "The AI interpretation could not be generated in time, but the extracted schema can still be reviewed manually."
        )