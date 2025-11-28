# extract_entities.py
import re
import unicodedata
from nltk import sent_tokenize, pos_tag, word_tokenize, everygrams
from nltk.corpus import wordnet, stopwords
from datetime import datetime
import calendar

# -----------------------------
# Helpers
# -----------------------------
def _norm_ascii_lower(s: str) -> str:
    """Normalize to ASCII (strip diacritics) and lowercase."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()


# -----------------------------
# Phone / Email
# -----------------------------
def get_number(text: str):
    """
    Return list of phone numbers found in text (best-effort, deduped).
    Filters out year-like tails (1900–2100).
    """
    pattern = re.compile(
        r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)'
    )
    pt = pattern.findall(text or "")
    # normalize and keep reasonable lengths
    pt = [re.sub(r"[,.]", "", p) for p in pt if len(re.sub(r"[()\-\s+.,]", "", p)) > 9]
    pt = [re.sub(r"\D$", "", p).strip() for p in pt]
    pt = [p for p in pt if len(re.sub(r"\D", "", p)) <= 15]

    cleaned = []
    for ph in pt:
        if len(ph.split("-")) > 3:
            continue
        bad = False
        for seg in ph.split("-"):
            seg = seg.strip()
            if seg[-4:].isdigit():
                try:
                    if 1900 <= int(seg[-4:]) <= 2100:
                        bad = True
                        break
                except Exception:
                    pass
        if not bad:
            cleaned.append(ph)

    return list(set(cleaned))


def get_email(text: str):
    """Return list of emails found in text."""
    r = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+")
    return r.findall(text or "")


def rm_number(text: str):
    """Remove detected phone numbers from text."""
    if not text:
        return text
    try:
        pattern = re.compile(
            r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)'
        )
        pt = pattern.findall(text)
        pt = [re.sub(r"[,.]", "", p) for p in pt if len(re.sub(r"[()\-\s+.,]", "", p)) > 9]
        pt = [re.sub(r"\D$", "", p).strip() for p in pt]
        pt = [p for p in pt if len(re.sub(r"\D", "", p)) <= 15]

        cleaned = []
        for ph in pt:
            if len(ph.split("-")) > 3:
                continue
            bad = False
            for seg in ph.split("-"):
                seg = seg.strip()
                if seg[-4:].isdigit():
                    try:
                        if 1900 <= int(seg[-4:]) <= 2100:
                            bad = True
                            break
                    except Exception:
                        pass
            if not bad:
                cleaned.append(ph)

        for ph in set(cleaned):
            text = text.replace(ph, " ")
        return text
    except Exception:
        return text


def rm_email(text: str):
    """Remove emails from text."""
    if not text:
        return text
    try:
        pattern = re.compile(r"[\w\.-]+@[\w\.-]+")
        for em in set(pattern.findall(text)):
            text = text.replace(em, " ")
        return text
    except Exception:
        return text


# -----------------------------
# Name (very lightweight heuristic)
# -----------------------------
def get_name(text: str):
    """
    Extract a candidate name using a simple heuristic:
    nouns not in WordNet from early sentences (first plausible token).
    """
    if not text:
        return ""
    sentences = sent_tokenize(text)
    tokens_by_sent = [word_tokenize(s) for s in sentences[:3]]  # focus on the top
    tagged = [pos_tag(tokens) for tokens in tokens_by_sent]

    nouns = []
    for sent in tagged:
        for tok, tag in sent:
            if re.match(r"NN.*", tag):
                nouns.append(tok)

    cands = [n for n in nouns if not wordnet.synsets(n)]
    return " ".join(cands[:1]) if cands else ""


# -----------------------------
# Skills
# -----------------------------
def get_skills(text: str, skills):
    """
    Return a set of matched skills (unigrams + bigrams/trigrams)
    against a lowercased vocabulary `skills`.
    """
    sw = set(stopwords.words("english"))
    tokens = word_tokenize(text or "")
    ft = [w for w in tokens if w.isalpha()]
    ft = [w.lower() for w in ft if w.lower() not in sw]

    ngrams = list(map(" ".join, everygrams(ft, 2, 3)))
    found = set()

    for tok in ft:
        if tok in skills:
            found.add(tok)
    for ng in ngrams:
        if ng in skills:
            found.add(ng)
    return found


# -----------------------------
# Romania-specific extras
# -----------------------------
RO_CITIES = [
    "București", "Cluj", "Cluj-Napoca", "Iași", "Timisoara", "Timișoara", "Constanța",
    "Brașov", "Sibiu", "Craiova", "Oradea", "Galați", "Ploiești", "Arad", "Bacău",
    "Brăila", "Buzău", "Satu Mare", "Baia Mare", "Suceava", "Pitești", "Piatra Neamț",
    "Târgu Mureș", "Botoșani", "Focșani", "Alba Iulia", "Deva", "Târgu Jiu"
]


import re, unicodedata

def _ascii(s: str) -> str:
    return unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()

RO_CITIES_CANON = [
    "Bucuresti","Cluj-Napoca","Iasi","Timisoara","Constanta","Brasov","Sibiu",
    "Craiova","Oradea","Galati","Ploiesti","Arad","Bacau","Braila","Buzau",
    "Satu Mare","Baia Mare","Suceava","Pitesti","Piatra Neamt","Targu Mures",
    "Botosani","Focsani","Alba Iulia","Deva","Targu Jiu"
]
CITY_INDEX = { _ascii(c): c for c in RO_CITIES_CANON }

def get_location(text: str) -> str:
    t = _ascii(text)
    for key, canon in CITY_INDEX.items():
        if re.search(rf"\b{re.escape(key)}\b", t):
            return canon  # already ASCII (Galati)
    m = re.search(r"(?:city|oras|orasul|localitate)\s*[:\-]\s*([A-Za-z \-]+)", t, re.I)
    if m:
        guess = _ascii(m.group(1)).strip()
        return CITY_INDEX.get(guess, m.group(1).strip().title())
    return ""


# helper: month name/abbr mapping
_MON = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MON.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

def _parse_month_year(tok: str):
    """
    Parse 'Dec 2018', 'December 2018', '2018' → (year, month)
    Falls back to (year, 1) if month missing/unknown.
    """
    tok = tok.strip().replace(",", " ")
    parts = tok.split()
    if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) == 4:
        return int(parts[0]), 1
    if len(parts) >= 2:
        m = _MON.get(parts[0].lower())
        y = None
        for p in parts[1:]:
            if p.isdigit() and len(p) == 4:
                y = int(p); break
        if m and y:
            return y, m
    # last resort: any 4-digit year
    y = re.search(r"\b(19|20)\d{2}\b", tok)
    return (int(y.group()), 1) if y else None

def get_years_experience(text: str) -> str:
    """
    Max tenure in years from:
      - explicit 'X years / ani'
      - date ranges like 'Dec 2018 – Present', '2016-2020', '2015 to 2019'
    """
    if not text:
        return "0"

    # 1) explicit 'X years/ani'
    years = [int(x) for x in re.findall(r"\b(\d{1,2})\s*(?:years?|ani|an)\b", text, re.I)]

    # 2) date ranges
    # normalize dashes
    t = re.sub(r"[–—−]", "-", text)

    # patterns: 'Dec 2018 - Present', 'December 2018 - June 2021', '2016-2020'
    pat_words = re.findall(r"([A-Za-z]{3,9}\s+\d{4})\s*-\s*(Present|[A-Za-z]{3,9}\s+\d{4}|\d{4})", t, re.I)
    pat_years = re.findall(r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2}|Present)\b", t, re.I)

    ranges = []
    for a, b in pat_words:
        s = _parse_month_year(a)
        if b.lower() == "present":
            e = (datetime.now().year, datetime.now().month)
        else:
            e = _parse_month_year(b) or (int(b), 1)
        if s and e:
            ranges.append((s, e))

    for a, b in pat_years:
        s = (int(a), 1)
        e = (datetime.now().year, datetime.now().month) if b.lower() == "present" else (int(b), 1)
        ranges.append((s, e))

    # compute years for each range
    tenures = []
    for (ys, ms), (ye, me) in ranges:
        try:
            ds = datetime(ys, ms, 1)
            de = datetime(ye, me, 1)
            months = max(0, (de.year - ds.year) * 12 + (de.month - ds.month))
            tenures.append(round(months / 12, 1))  # 1 decimal
        except Exception:
            pass

    all_vals = years + [int(round(x)) for x in tenures]  # round to int years
    return str(max(all_vals)) if all_vals else "0"


def get_education(text: str):
    """Light heuristic for degree keywords (deduped, order-preserving)."""
    if not text:
        return ""
    EDU_PAT = r"(MD|Medic|Doctor of Medicine|PhD|Master(?:'s)?|MSc|MS|MBA|Bachelor(?:'s)?|BSc|BS|Residency|Rezidențiat|Rezidentiat)"
    hits = re.findall(EDU_PAT, text, re.I)
    seen, ordered = set(), []
    for h in hits:
        key = h.strip().lower()
        if key not in seen:
            seen.add(key)
            ordered.append(h.strip())
    return ", ".join(ordered)
