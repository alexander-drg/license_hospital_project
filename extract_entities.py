import re
from nltk import sent_tokenize, pos_tag, word_tokenize, everygrams
from nltk.corpus import wordnet, stopwords

def get_number(text):
    pattern = re.compile(r'([+(]?\d+[)\-]?\s*[(]?\d{2,}[)\-]?\s*\d{2,}\s*\d*\s*\d*)')
    pt = pattern.findall(text or "")
    pt = [re.sub(r'[,.]', '', a) for a in pt if len(re.sub(r'[()\-.,\s+]', '', a))>9]
    pt = [re.sub(r'\D$', '', a).strip() for a in pt]
    pt = [a for a in pt if len(re.sub(r'\D','',a)) <= 15]
    for a in list(pt):
        if len(a.split('-'))>3: continue
        for x in a.split("-"):
            try:
                if x.strip()[-4:].isdigit() and int(x.strip()[-4:]) in range(1900,2100):
                    pt.remove(a)
            except: pass
    return list(set(pt)) if pt else []

def get_email(text):
    r = re.compile(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+')
    return r.findall(text or "")

def rm_number(text):
    try:
        for n in get_number(text or ""): text = text.replace(n, " ")
        return text
    except: return text

def rm_email(text):
    try:
        for e in get_email(text or ""): text = text.replace(e, " ")
        return text
    except: return text

def get_name(text):
    sents = sent_tokenize(text or "")
    words = [pos_tag(word_tokenize(s)) for s in sents]
    nouns = []
    for w in words:
        for tok in w:
            if re.match('[NN.*]', tok[1]): nouns.append(tok[0])
    cands = [n for n in nouns if not wordnet.synsets(n)]
    return ' '.join(cands[:1])

def get_skills(text, skills):
    sw = set(stopwords.words('english'))
    tokens = word_tokenize(text or "")
    ft = [w for w in tokens if w.isalpha() and w.lower() not in sw]
    ngrams = list(map(' '.join, everygrams(ft, 2, 3)))
    out = set()
    for t in ft:
        if t.lower() in skills: out.add(t.lower())
    for ng in ngrams:
        if ng.lower() in skills: out.add(ng.lower())
    return sorted(list(out))

RO_CITIES = ["Bucuresti","Cluj","Cluj-Napoca","Iasi","Timisoara","Constanta","Brasov","Sibiu","Craiova","Oradea","Galati","Ploiesti","Arad","Bacau","Braila","Buzau","Satu Mare","Baia Mare","Suceava","Pitesti","Piatra Neamt","Targu Mures","Botosani","Focsani","Alba Iulia","Deva","Targu Jiu"]

def get_location(text: str):
    t = text or ""
    for c in RO_CITIES:
        if re.search(rf"\b{re.escape(c)}\b", t, re.I): return c
    m = re.search(r"(?:city|oras|oraș)\s*[:\-]\s*([A-Za-z \-]+)", t, re.I)
    return m.group(1).strip() if m else ""

def get_years_experience(text: str):
    yrs = [int(x) for x in re.findall(r"\b(\d{1,2})\s*(?:years?|ani|an)\b", text or "", re.I)]
    return str(max(yrs)) if yrs else "0"

def get_education(text: str):
    EDU = r"(MD|Medic|Doctor of Medicine|PhD|Master(?:'s)?|MSc|MS|MBA|Bachelor(?:'s)?|BSc|BS|Residency|Rezidentiat)"
    hits = re.findall(EDU, text or "", re.I)
    seen, out = set(), []
    for h in hits:
        k = h.strip().lower()
        if k not in seen:
            seen.add(k); out.append(h)
    return ", ".join(out)
