# txt_to_features.py
from sklearn.feature_extraction.text import TfidfVectorizer

def txt_features(resume_texts, jd_texts):
    """
    Build a shared TF-IDF space for resumes + JD.
    Returns a sparse matrix with rows = [all resumes..., all jd texts...].
    """
    texts = list(resume_texts) + list(jd_texts)
    vec = TfidfVectorizer(min_df=1, stop_words="english")
    X = vec.fit_transform(texts)
    return X

def feats_reduce(X):
    """
    No-op dimensionality reduction placeholder.
    Keep this signature so older code continues to work.
    """
    return X
