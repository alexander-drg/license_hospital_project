# app/ranker.py

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .embeddings import Embedder
from .preprocess import (
    normalize_text,
    normalize_ro_en_text,
    split_list_field,
    skill_overlap,
)
from .competency import competency_match


class HybridRanker:
    """
    Two-stage recommender:
      1) Embed candidate CV + job descriptions, shortlist top-N by cosine.
      2) Cross-encoder re-ranks shortlist.
    Final score blends feature scores + cross-encoder score.

    This version also applies Romanian-English terminology normalization,
    so Romanian job descriptions can match English CVs more reliably.
    """

    def __init__(self, db, weights=None, shortlist_size=50, device=None):
        self.db = db
        self.shortlist_size = shortlist_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.w = weights or {
            "emb": 0.45,
            "skills": 0.18,
            "salary": 0.10,
            "geo": 0.05,
            "schedule": 0.05,
            "comp": 0.17,
        }

        self.embedder = Embedder()

        # ---------------------------------------------------------
        # Precompute job cache
        # ---------------------------------------------------------
        self.jobs = list(self.db.jobs_iter())
        self.job_ids = [j["id"] for j in self.jobs]

        self.job_texts = [
            self._build_job_matching_text(j)
            for j in self.jobs
        ]

        self.job_embs = self.embedder.encode_texts(self.job_texts)
        self.id2idx = {jid: i for i, jid in enumerate(self.job_ids)}

        # ---------------------------------------------------------
        # Cross-encoder for reranking
        # ---------------------------------------------------------
        ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.ce_tok = AutoTokenizer.from_pretrained(ce_model_name)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(ce_model_name).to(self.device)
        self.ce_model.eval()

    # ---------------------------------------------------------
    # Text builders
    # ---------------------------------------------------------
    def _build_candidate_matching_text(self, candidate: dict) -> str:
        """
        Builds a richer candidate text for matching.
        Romanian-English normalization is applied here.
        """
        parts = [
            candidate.get("name", ""),
            candidate.get("specialization", ""),
            candidate.get("skills", ""),
            candidate.get("education", ""),
            candidate.get("summary", ""),
            candidate.get("profile_text", ""),
        ]

        text = "\n".join([str(p) for p in parts if p])
        text = normalize_ro_en_text(text)
        text = normalize_text(text)

        return text

    def _build_job_matching_text(self, job: dict) -> str:
        """
        Builds a richer job text for matching.
        Romanian-English normalization is applied here.
        """
        parts = [
            job.get("title", ""),
            job.get("specialization", ""),
            job.get("seniority", ""),
            job.get("must_have_skills", ""),
            job.get("nice_to_have_skills", ""),
            job.get("summary", ""),
            job.get("description_text", ""),
            job.get("contract_type", ""),
            job.get("shift", ""),
            job.get("city", ""),
        ]

        text = "\n".join([str(p) for p in parts if p])
        text = normalize_ro_en_text(text)
        text = normalize_text(text)

        return text

    # ---------------------------------------------------------
    # Feature scorers
    # ---------------------------------------------------------
    def _salary_score(self, cand_min, job_min, job_max):
        try:
            c = int(cand_min or 0)
            lo = int(job_min or 0)
            hi = int(job_max or 0)
        except Exception:
            return 0.0

        if lo and hi and lo <= c <= hi:
            return 1.0

        if lo and c < lo:
            return max(0.0, 1 - (lo - c) / max(1, int(0.5 * lo)))

        if hi and c > hi:
            return max(0.0, 1 - (c - hi) / max(1, int(0.5 * hi)))

        return 0.0

    def _schedule_score(self, cand_pref, job_shift):
        if not cand_pref:
            return 0.5

        return 1.0 if cand_pref.lower() in (job_shift or "").lower() else 0.0

    def _geo_score(self, cand_city, job_city):
        if not cand_city or not job_city:
            return 0.5

        cand_city = cand_city.strip().lower()
        job_city = job_city.strip().lower()

        return 1.0 if cand_city == job_city else 0.5

    # ---------------------------------------------------------
    # Cross-encoder re-ranker
    # ---------------------------------------------------------
    def _rerank_cross_encoder(self, cv_text, jobs_subset):
        """
        jobs_subset: list[dict]
        returns: dict job_id -> score in [0,1]
        """
        if not jobs_subset:
            return {}

        pairs = [
            (
                cv_text,
                self._build_job_matching_text(j)
            )
            for j in jobs_subset
        ]

        enc = self.ce_tok.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.ce_model(**enc).logits.squeeze(-1)
            scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()

        return {
            j["id"]: float(s)
            for j, s in zip(jobs_subset, scores)
        }

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def recommend(self, candidate_id, top_k=10, with_reasons=False):
        candidate = self.db.get_candidate(candidate_id)

        if not candidate:
            return []

        # Per-candidate weights if learner exists
        w = getattr(getattr(self.db, "learner", None), "weights_for", None)
        W = w(candidate_id) if callable(w) else self.w

        # Candidate matching text
        cv_text = self._build_candidate_matching_text(candidate)
        c_emb = self.embedder.encode_texts([cv_text])[0]

        # Skills are normalized too
        candidate_skills_text = normalize_ro_en_text(candidate.get("skills", ""))
        c_skills = split_list_field(candidate_skills_text)

        # ---------------------------------------------------------
        # Stage 1: shortlist by embedding cosine
        # ---------------------------------------------------------
        if len(self.jobs) == 0:
            return []

        cosines = np.dot(self.job_embs, c_emb)

        idx_sorted = np.argsort(-cosines)[: min(self.shortlist_size, len(self.jobs))]
        shortlist = [self.jobs[i] for i in idx_sorted]

        # ---------------------------------------------------------
        # Stage 2: cross-encoder rerank
        # ---------------------------------------------------------
        ce_scores = self._rerank_cross_encoder(cv_text, shortlist)

        # ---------------------------------------------------------
        # Final scoring
        # ---------------------------------------------------------
        out = []

        for job in shortlist:
            jidx = self.id2idx[job["id"]]

            emb_s = float(cosines[jidx])

            job_skills_text = normalize_ro_en_text(job.get("must_have_skills", ""))
            job_skills = split_list_field(job_skills_text)

            skills_s = skill_overlap(job_skills, c_skills)

            salary_s = self._salary_score(
                candidate.get("salary_min", ""),
                job.get("salary_min", ""),
                job.get("salary_max", "")
            )

            geo_s = self._geo_score(
                candidate.get("city", ""),
                job.get("city", "")
            )

            sched_s = self._schedule_score(
                candidate.get("schedule_pref", ""),
                job.get("shift", "")
            )

            normalized_job_requirements = normalize_ro_en_text(job.get("must_have_skills", ""))
            normalized_candidate_profile = normalize_ro_en_text(candidate.get("profile_text", ""))

            comp_s = competency_match(
                normalized_job_requirements,
                normalized_candidate_profile
            )

            linear_score = (
                W["emb"] * emb_s
                + W["skills"] * skills_s
                + W["salary"] * salary_s
                + W["geo"] * geo_s
                + W["schedule"] * sched_s
                + W["comp"] * comp_s
            )

            ce = ce_scores.get(job["id"], 0.5)

            final_score = 0.8 * float(linear_score) + 0.2 * float(ce)

            item = {
                "job_id": job["id"],
                "title": job.get("title", ""),
                "score": round(final_score, 4),
            }

            if with_reasons:
                item["reasons"] = {
                    "emb": round(emb_s, 3),
                    "skills": round(skills_s, 3),
                    "salary": round(salary_s, 3),
                    "geo": round(geo_s, 3),
                    "schedule": round(sched_s, 3),
                    "comp": round(comp_s, 3),
                    "ce": round(ce, 3),
                }

            out.append(item)

        out.sort(key=lambda x: x["score"], reverse=True)

        return out[:top_k]