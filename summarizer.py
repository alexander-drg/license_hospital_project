# summarizer.py
import re
from textwrap import shorten
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')

def _sentences(text: str):
    if not text:
        return []
    parts = [p.strip(" -*•\t") for p in _SENT_SPLIT.split(text)]
    return [s for s in parts if 4 <= len(s) <= 300]

def summarize_against_jd(cv_text: str, jd_text: str, max_sent: int = 3):
    """
    Pick the top N sentences from the CV that are most relevant to the JD.
    Relevance = cosine on a TF-IDF space using (JD + each sentence).
    Returns a list of short strings (no bullets, caller can join with ' • ').
    """
    sents = _sentences(cv_text)
    if not sents:
        return []
    docs = [jd_text] + sents
    vec = TfidfVectorizer(min_df=1, stop_words="english")
    X = vec.fit_transform(docs)
    jd_vec, sent_mat = X[0], X[1:]
    sims = (sent_mat @ jd_vec.T).toarray().ravel()
    order = np.argsort(-sims)[:max_sent]
    out = []
    for i in order:
        s = re.sub(r'\s+', ' ', sents[i]).strip()
        out.append(shorten(s, width=220, placeholder="…"))
    return out
