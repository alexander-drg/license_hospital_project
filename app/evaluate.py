import pandas as pd
from datetime import timedelta

def precision_at_k(recs, positives, k=5):
    ids = [r["job_id"] for r in recs[:k]]
    return sum(1 for x in ids if x in positives) / max(1,k)

def ndcg_at_k(recs, positives, k=10):
    import math
    dcg = sum((1.0 if r["job_id"] in positives else 0.0)/math.log2(i+2) for i,r in enumerate(recs[:k]))
    ideal = sum(1.0/math.log2(i+2) for i in range(min(len(positives),k)))
    return dcg / ideal if ideal>0 else 0.0

def retention_rate(interactions_df: pd.DataFrame, horizon_days=90):
    """Proxy: among 'hired' events, fraction that also have any positive event after H days."""
    if interactions_df.empty: return 0.0
    df = interactions_df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    hired = df[df["event"]=="hired"]
    kept = 0
    for _,h in hired.iterrows():
        cutoff = h["ts"] + timedelta(days=horizon_days)
        follow = df[(df.candidate_id==h.candidate_id) & (df.job_id==h.job_id) &
                    (df.ts>=cutoff) & (df.event.isin(["view","save","apply","hired"]))].any().any()
        kept += 1 if follow else 0
    return kept / max(1,len(hired))
