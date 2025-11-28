from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def simil(feats, p_resumes, p_jd):
    sim = cosine_similarity(feats[0:len(p_resumes)], feats[len(p_resumes):])
    cols = [f"JD {i}" for i in range(1, len(p_jd)+1)]
    return pd.DataFrame(sim, columns=cols)
