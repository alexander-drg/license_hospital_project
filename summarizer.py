# summarizer.py
from typing import List
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def summarize_against_jd(cv_text: str, jd_text: str, top_k: int = 3) -> List[str]:
    """
    Extractive summary: pick the sentences from the CV that are most similar
    to the job description using TF-IDF cosine similarity.
    """
    if not cv_text or not jd_text:
        return []

    sentences = [s.strip() for s in sent_tokenize(cv_text) if len(s.strip()) > 30]
    if not sentences:
        return []

    corpus = [jd_text] + sentences  # 0 = JD, rest = CV sentences
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(corpus)
    jd_vec = X[0:1]
    sent_vecs = X[1:]
    sims = cosine_similarity(sent_vecs, jd_vec).ravel()

    # top-k unique sentences by similarity
    order = sims.argsort()[::-1]
    picked = []
    used = set()
    for idx in order:
        s = sentences[idx]
        # avoid nearly-duplicates by sentence length/first chars heuristic
        key = (len(s), s[:40].lower())
        if key in used:
            continue
        used.add(key)
        picked.append(s)
        if len(picked) >= top_k:
            break
    return picked
