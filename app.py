# app.py
import os, re, json, uuid, unicodedata
from textwrap import shorten

import numpy as np
import pandas as pd
from flask import (
    Flask, flash, request, redirect, url_for,
    render_template, send_file, abort
)

# ---- local modules ----
from extract_txt import read_files
from txt_processing import preprocess
from txt_to_features import txt_features, feats_reduce
from model import simil
from extract_entities import (
    get_number, get_email, rm_email, rm_number,
    get_name, get_skills, get_location,
    get_years_experience, get_education
)
from summarizer import summarize_against_jd

# ------------------ paths ------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
FILES_DIR      = os.path.join(BASE_DIR, "files")
UPLOAD_FOLDER  = os.path.join(FILES_DIR, "resumes")
DOWNLOAD_FOLDER= os.path.join(FILES_DIR, "outputs")
DATA_FOLDER    = os.path.join(BASE_DIR, "Data")
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|•|\n+')
HAS_EMAIL   = re.compile(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+')
HAS_PHONE   = re.compile(r'(?:\+?\d[\d\s().-]){7,}')
TOO_SHORT   = re.compile(r'^[\W_]{0,3}$')


os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# persist files
FILES_JSON   = os.path.join(UPLOAD_FOLDER, "files.json")
JD_STORE     = os.path.join(DOWNLOAD_FOLDER, "latest_jd.json")
LATEST_JSON  = os.path.join(DOWNLOAD_FOLDER, "latest_shortlist.json")
for _p in (JD_STORE, LATEST_JSON):
    if not os.path.exists(_p):
        with open(_p, "w", encoding="utf-8") as fh:
            fh.write("{}")

# ------------------ app ------------------
app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    DOWNLOAD_FOLDER=DOWNLOAD_FOLDER,
    DATA_FOLDER=DATA_FOLDER,
    SECRET_KEY="nani?!",
)
ALLOWED_EXTENSIONS = {"txt", "pdf", "doc", "docx"}

# ------------------ template filters ------------------
@app.template_filter("fmt")
def fmt(x):
    """Pretty print single field: list/set/tuple -> comma string; None -> ''."""
    if x is None:
        return ""
    if isinstance(x, (set, list, tuple)):
        return ", ".join(str(i) for i in x)
    return str(x)

@app.template_filter("bullets")
def bullets(s):
    """Split our ' • ' summary string into bullet lines."""
    if not s:
        return []
    return [p.strip() for p in str(s).split("•") if p.strip()]

# ------------------ helpers ------------------
def sentence_bullets(
    text: str,
    max_bullets: int = 6,
    min_per_bullet: int = 2,
    max_per_bullet: int = 3,
    width: int = 220
):
    """
    Turn a long blob of text into ~2–3 sentence bullets.
    - Filters out emails/phones/noise.
    - Wraps each bullet to a reasonable width.
    """
    if not text:
        return []

    # 1) split to sentences-ish
    parts = [p.strip(" -•\t") for p in SENT_SPLIT.split(str(text)) if p and not TOO_SHORT.match(p)]

    # 2) filter obvious noise (emails/phones)
    sent = [s for s in parts if not HAS_EMAIL.search(s) and not HAS_PHONE.search(s)]

    # 3) chunk 2–3 sentences per bullet
    bullets = []
    i = 0
    while i < len(sent) and len(bullets) < max_bullets:
        # choose how many sentences to pack in this bullet
        take = max_per_bullet
        # if near the end, you might only have 1–2 left; ensure at least min_per when possible
        remaining = len(sent) - i
        if remaining < min_per_bullet:
            take = remaining
        elif remaining < max_per_bullet:
            take = remaining
        # join and tidy
        chunk = " ".join(sent[i:i+take])
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if chunk:
            bullets.append(shorten(chunk, width=width, placeholder="…"))
        i += take

    return bullets
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _get_files():
    if os.path.exists(FILES_JSON):
        with open(FILES_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    return {}

def _save_files(files_map: dict):
    with open(FILES_JSON, "w", encoding="utf-8") as fh:
        json.dump(files_map, fh)

def _has_results() -> bool:
    try:
        return os.path.getsize(LATEST_JSON) > 2
    except Exception:
        return False

def asciiify(s):
    """Strip Romanian diacritics etc. -> ASCII-only string."""
    if s is None:
        return ""
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()

def flatten_join(x):
    """List/set/tuple -> 'a, b, c'. Strings pass through. None -> ''."""
    if x is None:
        return ""
    if isinstance(x, (set, list, tuple)):
        return ", ".join(str(i) for i in x)
    return str(x)

def sentence_bullets(text, max_sent=6, width=180):
    """Turn a long blob of text into <= max_sent short sentences for bullets."""
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+|•|\n+', str(text))
    out = []
    for s in parts:
        s = s.strip(" -•\t")
        if len(s) < 4:
            continue
        out.append(shorten(s, width=width, placeholder="…"))
        if len(out) >= max_sent:
            break
    return out

# ------------------ routes ------------------
@app.route("/", methods=["GET"])
def main_page():
    files = _get_files()
    jd_text = ""
    try:
        with open(JD_STORE, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
            jd_text = data.get("jd_text", "")
    except Exception:
        jd_text = ""
    return render_template("index.html", files=files, has_results=_has_results(), jd_text=jd_text)

@app.route("/", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    upload_files = request.files.getlist("file")
    if not upload_files:
        flash("No selected file")
        return redirect(request.url)

    files = _get_files()
    for file in upload_files:
        original_filename = file.filename
        if allowed_file(original_filename):
            ext = original_filename.rsplit(".", 1)[1].lower()
            filename = f"{uuid.uuid1()}.{ext}"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            files[filename] = original_filename

    _save_files(files)
    flash("Upload succeeded")
    return redirect(url_for("main_page"))

@app.route("/download/<code>", methods=["GET"])
def download(code):
    files = _get_files()
    if code in files:
        path = os.path.join(UPLOAD_FOLDER, code)
        if os.path.exists(path):
            return send_file(path)
    abort(404)

@app.route("/delete/<code>", methods=["POST"])
def delete(code):
    files = _get_files()
    if code in files:
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, code))
        except FileNotFoundError:
            pass
        del files[code]
        _save_files(files)
        flash("File deleted")
    return redirect(url_for("main_page"))

@app.route("/delete_all", methods=["POST"])
def delete_all():
    files = _get_files()
    for code in list(files.keys()):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, code))
        except FileNotFoundError:
            pass
        del files[code]
    _save_files(files)
    flash("All files deleted")
    return redirect(url_for("main_page"))

@app.route("/download_candidates", methods=["GET"])
def download_candidates():
    out_path = os.path.join(app.config["DOWNLOAD_FOLDER"], "Candidates.csv")
    if os.path.exists(out_path):
        return send_file(out_path, as_attachment=True)
    flash("No results yet")
    return redirect(url_for("main_page"))

@app.route("/results", methods=["GET"])
def results():
    if not _has_results():
        flash("No results yet")
        return redirect(url_for("main_page"))
    with open(LATEST_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    return render_template("results.html", cols=payload["columns"], rows=payload["rows"])

@app.route("/process", methods=["POST"])
def process():
    """Process JD + uploaded CVs, compute composite score, save CSV+JSON, stay on home."""
    import numpy as np
    import unicodedata
    import re
    from textwrap import shorten

    rawtext = request.form["rawtext"]

    # Persist JD so the textarea is prefilled next time
    try:
        with open(JD_STORE, "w", encoding="utf-8") as fh:
            json.dump({"jd_text": rawtext}, fh, ensure_ascii=False)
    except Exception:
        pass

    # ---------- small helpers (scoped to this route) ----------
    def first_item(x):
        """List/set/tuple -> first string; scalar -> str(x)."""
        if x is None:
            return ""
        if isinstance(x, (list, tuple, set)):
            for v in x:
                return str(v)
            return ""
        return str(x)

    def build_summary(cv_text: str,
                      jd_text: str,
                      max_bullets: int = 6,
                      min_per_bullet: int = 2,
                      max_per_bullet: int = 3,
                      width: int = 220):
        """
        Return ~2–3 sentence bullets from the CV text.
        Strips email/phone/links and location-ish noise, groups sentences per bullet.
        """
        txt = cv_text or ""

        # strip contact/links
        txt = re.sub(r'\S+@\S+', ' ', txt)                                  # emails
        txt = re.sub(r'(?:\+?\d[\d\s\-().]{6,})', ' ', txt)                 # phones
        txt = re.sub(r'https?://\S+|linkedin\.com/\S+', ' ', txt, flags=re.I)

        # drop location-ish words so summary doesn't repeat header
        ban_words = r'(phone|email|linkedin|address|city|location|bucurest[ii]|cluj|iasi|constanta|romania)'
        txt = re.sub(ban_words, ' ', txt, flags=re.I)

        # split into sentences-ish
        SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|•|\n+')
        parts = [p.strip(" -*•\t") for p in SENT_SPLIT.split(txt) if p and len(p.strip()) > 3]

        # filter obvious noise again
        HAS_EMAIL = re.compile(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+')
        HAS_PHONE = re.compile(r'(?:\+?\d[\d\s().-]){7,}')
        sentences = [s for s in parts if not HAS_EMAIL.search(s) and not HAS_PHONE.search(s)]

        # group 2–3 sentences per bullet
        bullets, i = [], 0
        while i < len(sentences) and len(bullets) < max_bullets:
            remaining = len(sentences) - i
            take = max_per_bullet if remaining >= max_per_bullet else remaining
            if remaining < min_per_bullet:
                take = remaining  # last bullet may be shorter
            chunk = " ".join(sentences[i:i+take])
            chunk = re.sub(r'\s+', ' ', chunk).strip()
            if chunk:
                bullets.append(shorten(chunk, width=width, placeholder="…"))
            i += take

        return bullets

    def skills_to_str(x):
        if isinstance(x, set):
            return ", ".join(sorted(list(x)))
        if isinstance(x, list):
            return ", ".join([str(i) for i in x])
        return str(x) if x is not None else ""

    def normalize(v):
        if isinstance(v, set):
            return sorted(list(v))
        if isinstance(v, tuple):
            return [normalize(x) for x in v]
        if hasattr(v, "item"):  # numpy scalar
            try:
                return v.item()
            except Exception:
                pass
        if isinstance(v, list):
            return [normalize(x) for x in v]
        return v

    def norm_ascii(s: str) -> str:
        return unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()

    # ---------- 1) Read & preprocess ----------
    resumetxt   = read_files(UPLOAD_FOLDER)
    p_resumetxt = preprocess(resumetxt)
    p_jdtxt     = preprocess([rawtext])

    feats     = txt_features(p_resumetxt, p_jdtxt)
    feats_red = feats_reduce(feats)

    # ---------- 2) Cosine similarity ----------
    df   = simil(feats_red, p_resumetxt, p_jdtxt)        # has column "JD 1"
    base = pd.DataFrame({"Original Resume": resumetxt})
    dt   = pd.concat([df, base], axis=1)

    # ---------- 3) Contacts + cleaned text ----------
    dt["Phone No."]        = dt["Original Resume"].apply(get_number)
    dt["E-Mail ID"]        = dt["Original Resume"].apply(get_email)
    dt["Original"]         = dt["Original Resume"].apply(rm_number).apply(rm_email)
    dt["Candidate's Name"] = dt["Original"].apply(get_name)

    # normalize e-mail / phone to plain strings (no brackets)
    dt["E-Mail ID"] = dt["E-Mail ID"].apply(first_item)
    dt["Phone No."] = dt["Phone No."].apply(first_item)

    # ---------- 4) Skills / extras ----------
    skills_df   = pd.read_csv(os.path.join(DATA_FOLDER, "skill_red.csv"))
    skill_vocab = [str(z).lower() for z in skills_df.values.flatten().tolist()]
    dt["Skills"]           = dt["Original"].apply(lambda x: get_skills(x, skill_vocab))
    dt["Location"]         = dt["Original"].apply(get_location)
    dt["Experience (yrs)"] = dt["Original"].apply(get_years_experience)
    dt["Education"]        = dt["Original"].apply(get_education)

    # 2–3 sentence bullets (don’t include phone/city/etc.)
    dt["SummaryLines"] = [
        build_summary(cv, rawtext, max_bullets=6, min_per_bullet=2, max_per_bullet=3, width=220)
        for cv in dt["Original Resume"].tolist()
    ]

    # ---------- 5) Parse JD for scoring ----------
    def parse_jd(text: str):
        jd_city  = get_location(text)
        try:
            jd_years = int(get_years_experience(text) or "0")
        except Exception:
            jd_years = 0
        jd_tokens = set(w.lower() for w in re.findall(r"[A-Za-zăâîșț\-]+", text))
        jd_skills = set([s for s in skill_vocab if s in jd_tokens])
        edu_req   = set(m.lower() for m in re.findall(
            r"\b(Medic|MD|Rezidentiat|Rezidențiat|Master|MSc|MS|PhD|Doctor|Bachelor|BSc|BS)\b", text, re.I))
        return jd_city, jd_years, jd_skills, edu_req

    jd_city, jd_years, jd_skills, edu_req = parse_jd(rawtext)

    # ---------- 6) Component scores (0..1) ----------
    cos = pd.to_numeric(dt["JD 1"], errors="coerce").fillna(0.0).clip(0, 1)

    def skills_overlap(cv_sk):
        cv = set([str(x).lower() for x in (cv_sk or [])]) if isinstance(cv_sk, (set, list)) else set()
        if not jd_skills:
            return 0.5
        return len(cv & jd_skills) / max(1, len(jd_skills))
    skills_score = dt["Skills"].apply(skills_overlap)

    def geo_match(c_city):
        if not c_city:
            return 0.0
        if jd_city and norm_ascii(c_city) == norm_ascii(jd_city):
            return 1.0
        return 0.6
    geo_score = dt["Location"].apply(geo_match)

    def exp_match(c_years):
        try:
            y = float(c_years)
        except Exception:
            y = 0.0
        if jd_years <= 0:
            return 0.5
        return min(1.0, y / jd_years)
    exp_score = dt["Experience (yrs)"].apply(exp_match)

    def edu_match(c_edu):
        c = str(c_edu or "").lower()
        if not edu_req:
            return 0.5
        return 1.0 if any(tok in c for tok in edu_req) else 0.0
    edu_score = dt["Education"].apply(edu_match)

    # Weighted blend -> 1..100
    w_cos, w_sk, w_geo, w_exp, w_edu = 0.50, 0.25, 0.10, 0.10, 0.05
    score01 = (w_cos*cos + w_sk*skills_score + w_geo*geo_score + w_exp*exp_score + w_edu*edu_score)
    dt["Score"] = np.rint(score01.clip(0, 1) * 100).astype(int)

    # ---------- 7) Display / export ----------
    display_cols = [
        "Score",
        "Candidate's Name", "E-Mail ID", "Phone No.",
        "Location", "Experience (yrs)", "Education", "Skills", "SummaryLines",
        "CV File", "CV"
    ]
    for c in display_cols:
        if c not in dt.columns:
            dt[c] = ""
    sorted_dt = dt[display_cols].sort_values(by=["Score"], ascending=False)

    # CSV (pretty skills string)
    table_for_csv = sorted_dt.copy()
    table_for_csv["Skills"] = table_for_csv["Skills"].apply(skills_to_str)
    out_path = os.path.join(DOWNLOAD_FOLDER, "Candidates.csv")
    table_for_csv.to_csv(out_path, index=False)

    # JSON for /results
    rows      = sorted_dt.to_dict(orient="records")
    norm_rows = [{k: normalize(v) for k, v in r.items()} for r in rows]
    payload   = {"columns": list(sorted_dt.columns), "rows": norm_rows}
    with open(LATEST_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)

    flash("Job processed. View results or download CSV.")
    return redirect(url_for("main_page"))


# ------------------ main ------------------
if __name__ == "__main__":
    app.run(port=8080, debug=False)
