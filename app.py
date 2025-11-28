import os, re, json, uuid, unicodedata
from textwrap import shorten
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_file, flash, abort

from extract_txt import read_files
from txt_processing import preprocess
from txt_to_features import txt_features, feats_reduce
from model import simil
from extract_entities import (
    get_number, get_email, rm_email, rm_number,
    get_name, get_skills, get_location,
    get_years_experience, get_education
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, "files")
UPLOAD_FOLDER = os.path.join(FILES_DIR, "resumes")
OUTPUT_FOLDER = os.path.join(FILES_DIR, "outputs")
DATA_FOLDER = os.path.join(BASE_DIR, "Data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

LATEST_JSON = os.path.join(OUTPUT_FOLDER, "latest_shortlist.json")
JD_STORE = os.path.join(OUTPUT_FOLDER, "latest_jd.json")

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    DOWNLOAD_FOLDER=OUTPUT_FOLDER,
    DATA_FOLDER=DATA_FOLDER,
    SECRET_KEY="nani?!",
)

ALLOWED = {"pdf", "docx", "doc", "txt"}

# ---------- helpers ----------
def allowed_file(fn): return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED

def _files_map():
    p = os.path.join(UPLOAD_FOLDER, "files.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh: return json.load(fh)
    return {}

def _save_files(m): 
    with open(os.path.join(UPLOAD_FOLDER, "files.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=2)

def _has_results(): return os.path.exists(LATEST_JSON)

def asciiify(s):
    if s is None: return ""
    return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode()

def first_item(x):
    if x is None: return ""
    if isinstance(x, (list, tuple, set)):
        for v in x: return str(v)
        return ""
    return str(x)

def build_summary(cv_text: str, jd_text: str, max_bullets=6, min_per=2, max_per=3, width=220):
    txt = cv_text or ""
    txt = re.sub(r'\S+@\S+', ' ', txt)
    txt = re.sub(r'(?:\+?\d[\d\s\-().]{6,})', ' ', txt)
    txt = re.sub(r'https?://\S+|linkedin\.com/\S+', ' ', txt, flags=re.I)
    ban_words = r'(phone|email|linkedin|address|city|location|bucurest[ii]|cluj|iasi|constanta|romania)'
    txt = re.sub(ban_words, ' ', txt, flags=re.I)

    SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|•|\n+')
    parts = [p.strip(" -*•\t") for p in SENT_SPLIT.split(txt) if p and len(p.strip()) > 3]
    HAS_EMAIL = re.compile(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+')
    HAS_PHONE = re.compile(r'(?:\+?\d[\d\s().-]){7,}')
    sentences = [s for s in parts if not HAS_EMAIL.search(s) and not HAS_PHONE.search(s)]

    bullets, i = [], 0
    while i < len(sentences) and len(bullets) < max_bullets:
        remaining = len(sentences) - i
        take = max_per if remaining >= max_per else remaining
        if remaining < min_per: take = remaining
        chunk = " ".join(sentences[i:i+take])
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if chunk:
            bullets.append(shorten(chunk, width=width, placeholder="…"))
        i += take
    return bullets

# ---------- routes ----------
@app.route("/", methods=["GET"])
def index():
    files = _files_map()
    jd_text = ""
    try:
        if os.path.exists(JD_STORE):
            with open(JD_STORE, encoding="utf-8") as fh:
                jd_text = (json.load(fh) or {}).get("jd_text","")
    except Exception:
        jd_text = ""
    return render_template("index.html", files=files, has_results=_has_results(), jd_text=jd_text)

@app.route("/", methods=["POST"])
def upload():
    if "file" not in request.files: 
        flash("No file part"); return redirect(url_for("index"))
    files_in = request.files.getlist("file")
    if not files_in: 
        flash("No selected file"); return redirect(url_for("index"))

    files = _files_map()
    for f in files_in:
        if not f.filename: continue
        if not allowed_file(f.filename): continue
        ext = f.filename.rsplit(".",1)[1].lower()
        code = f"{uuid.uuid4().hex[:12]}.{ext}"
        f.save(os.path.join(UPLOAD_FOLDER, code))
        files[code] = f.filename
    _save_files(files)
    flash("Upload succeeded")
    return redirect(url_for("index"))

@app.route("/download/<code>")
def download(code):
    files = _files_map()
    if code in files:
        path = os.path.join(UPLOAD_FOLDER, code)
        if os.path.exists(path): return send_file(path)
    abort(404)

@app.route("/delete/<code>", methods=["POST"])
def delete(code):
    files = _files_map()
    if code in files:
        try: os.remove(os.path.join(UPLOAD_FOLDER, code))
        except FileNotFoundError: pass
        del files[code]; _save_files(files)
        flash("File deleted")
    return redirect(url_for("index"))

@app.route("/delete_all", methods=["POST"])
def delete_all():
    files = _files_map()
    for code in list(files.keys()):
        try: os.remove(os.path.join(UPLOAD_FOLDER, code))
        except FileNotFoundError: pass
        del files[code]
    _save_files(files)
    flash("All files deleted")
    return redirect(url_for("index"))

@app.route("/download_candidates")
def download_candidates():
    p = os.path.join(OUTPUT_FOLDER, "Candidates.csv")
    if os.path.exists(p): return send_file(p, as_attachment=True)
    flash("No results yet"); return redirect(url_for("index"))

@app.route("/results")
def results():
    if not os.path.exists(LATEST_JSON):
        flash("No results yet"); return redirect(url_for("index"))
    with open(LATEST_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    return render_template("results.html", cols=payload["columns"], rows=payload["rows"])

@app.route("/process", methods=["POST"])
def process():
    rawtext = request.form["rawtext"].strip()

    # persist JD
    try:
        with open(JD_STORE, "w", encoding="utf-8") as fh:
            json.dump({"jd_text": rawtext}, fh, ensure_ascii=False)
    except Exception: pass

    # read CVs
    resumes = read_files(UPLOAD_FOLDER)
    p_resumes = preprocess(resumes)
    p_jd = preprocess([rawtext])

    feats = txt_features(p_resumes, p_jd)
    feats_red = feats_reduce(feats)

    df = simil(feats_red, p_resumes, p_jd)  # has "JD 1"
    base = pd.DataFrame({"Original Resume": resumes})
    dt = pd.concat([df, base], axis=1)

    # contacts / identity
    dt["Phone No."]        = dt["Original Resume"].apply(get_number).apply(first_item)
    dt["E-Mail ID"]        = dt["Original Resume"].apply(get_email).apply(first_item)
    dt["Original"]         = dt["Original Resume"].apply(rm_number).apply(rm_email)
    dt["Candidate's Name"] = dt["Original"].apply(get_name)

    # skills/extras
    skills_csv = os.path.join(DATA_FOLDER, "skill_red.csv")
    skill_vocab = []
    if os.path.exists(skills_csv):
        s = pd.read_csv(skills_csv).values.flatten().tolist()
        skill_vocab = [str(z).lower() for z in s]
    dt["Skills"]           = dt["Original"].apply(lambda x: get_skills(x, skill_vocab))
    dt["Location"]         = dt["Original"].apply(get_location)
    dt["Experience (yrs)"] = dt["Original"].apply(get_years_experience)
    dt["Education"]        = dt["Original"].apply(get_education)

    # summaries (2–3 sentences per bullet)
    dt["SummaryLines"] = [build_summary(cv, rawtext, max_bullets=6, min_per=2, max_per=3, width=220)
                          for cv in dt["Original Resume"].tolist()]

    # parse JD for scoring
    def parse_jd(text: str):
        jd_city  = get_location(text)
        try: jd_years = int(get_years_experience(text) or "0")
        except Exception: jd_years = 0
        toks = set(w.lower() for w in re.findall(r"[A-Za-zăâîșț\-]+", text))
        jd_sk = set([s for s in skill_vocab if s in toks])
        edu_req = set(m.lower() for m in re.findall(
            r"\b(Medic|MD|Rezidentiat|Rezidențiat|Master|MSc|MS|PhD|Doctor|Bachelor|BSc|BS)\b", text, re.I))
        return jd_city, jd_years, jd_sk, edu_req

    jd_city, jd_years, jd_skills, edu_req = parse_jd(rawtext)

    cos = pd.to_numeric(dt["JD 1"], errors="coerce").fillna(0.0).clip(0, 1)

    def skills_overlap(cv_sk):
        cv = set([str(x).lower() for x in (cv_sk or [])]) if isinstance(cv_sk, (set, list)) else set()
        if not jd_skills: return 0.5
        return len(cv & jd_skills) / max(1, len(jd_skills))
    skills_score = dt["Skills"].apply(skills_overlap)

    def norm_ascii(s):
        return unicodedata.normalize("NFD", s or "").encode("ascii","ignore").decode().lower()

    def geo_match(c_city):
        if not c_city: return 0.0
        if jd_city and norm_ascii(c_city) == norm_ascii(jd_city): return 1.0
        return 0.6
    geo_score = dt["Location"].apply(geo_match)

    def exp_match(c_years):
        try: y = float(c_years)
        except Exception: y = 0.0
        if jd_years <= 0: return 0.5
        return min(1.0, y / jd_years)
    exp_score = dt["Experience (yrs)"].apply(exp_match)

    def edu_match(c_edu):
        c = str(c_edu or "").lower()
        if not edu_req: return 0.5
        return 1.0 if any(tok in c for tok in edu_req) else 0.0
    edu_score = dt["Education"].apply(edu_match)

    w_cos, w_sk, w_geo, w_exp, w_edu = 0.50, 0.25, 0.10, 0.10, 0.05
    score01 = (w_cos*cos + w_sk*skills_score + w_geo*geo_score + w_exp*exp_score + w_edu*edu_score)
    dt["Score"] = np.rint(score01.clip(0, 1) * 100).astype(int)

    # expose which file belongs to which row (for "View CV")
    files_map = _files_map()
    codes_by_resume = []
    for code, orig in files_map.items():
        try:
            path = os.path.join(UPLOAD_FOLDER, code)
            with open(path, "rb") as f: pass
        except Exception: pass
    # naive: attach first code filename for each row based on order
    # (better: hash file contents; for now order is sufficient)
    code_list = list(files_map.keys())
    dt["CV Code"] = pd.Series(code_list[:len(dt)] + [""]*(len(dt)-len(code_list)))

    # display/export
    display_cols = [
        "Score",
        "Candidate's Name", "E-Mail ID", "Phone No.",
        "Location", "Experience (yrs)", "Education", "Skills", "SummaryLines", "CV Code"
    ]
    for c in display_cols:
        if c not in dt.columns: dt[c] = ""
    sorted_dt = dt[display_cols].sort_values(by=["Score"], ascending=False)

    # CSV (pretty skills)
    def skills_to_str(x):
        if isinstance(x, set): return ", ".join(sorted(list(x)))
        if isinstance(x, list): return ", ".join([str(i) for i in x])
        return str(x) if x is not None else ""
    table_for_csv = sorted_dt.copy()
    table_for_csv["Skills"] = table_for_csv["Skills"].apply(skills_to_str)
    out_csv = os.path.join(OUTPUT_FOLDER, "Candidates.csv")
    table_for_csv.drop(columns=["SummaryLines"]).to_csv(out_csv, index=False)

    # JSON for UI
    def norm(v):
        if isinstance(v, set): return sorted(list(v))
        if isinstance(v, tuple): return [norm(x) for x in v]
        if hasattr(v, "item"):
            try: return v.item()
            except Exception: pass
        if isinstance(v, list): return [norm(x) for x in v]
        return v
    rows = sorted_dt.to_dict(orient="records")
    payload = {"columns": list(sorted_dt.columns), "rows": [{k: norm(v) for k,v in r.items()} for r in rows]}
    with open(LATEST_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    flash("Job processed. View results or download CSV.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(port=8080, debug=False)
