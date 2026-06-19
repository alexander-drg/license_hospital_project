from collections import defaultdict

EVENT_GAIN = {"view":0.05, "save":0.3, "apply":0.6, "reject":-0.4, "hired":1.0}

class SimpleLearner:
    def __init__(self, ranker, lr=0.05, decay=0.98):
        self.ranker = ranker
        self.lr = lr
        self.decay = decay
        self.user_w = defaultdict(lambda: dict(self.ranker.w))  # per-candidate copy

    def weights_for(self, cid):
        return self.user_w[cid]

    def update_from_feedback(self, fb):
        cid, jid, ev = fb["candidate_id"], fb["job_id"], (fb.get("event") or "").lower()
        gain = EVENT_GAIN.get(ev)
        if gain is None: return
        recs = self.ranker.recommend(cid, top_k=50, with_reasons=True)
        rs = next((r["reasons"] for r in recs if r["job_id"]==jid), None)
        if not rs: return
        w = self.user_w[cid]
        for k in ["emb","skills","salary","geo","schedule","comp"]:
            w[k] = max(0.0, w[k]*self.decay + self.lr*gain*rs.get(k,0.0))
        s = sum(w.values()) or 1.0
        for k in w: w[k] /= s
