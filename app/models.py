import json
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

CANDIDATES_JSON = DATA / "candidates_store.json"
JOBS_JSON = DATA / "jobs_store.json"
RESULTS_JSON = DATA / "results_store.json"

CANDIDATE_COLUMNS = [
    "id",
    "name",
    "specialization",
    "years_exp",
    "certs",
    "city",
    "salary_min",
    "schedule_pref",
    "skills",
    "profile_text",
    "email",
    "phone",
    "education",
    "summary",
    "source_filename",
    "domain_score",
    "is_medical_related",
]

JOB_COLUMNS = [
    "id",
    "title",
    "specialization",
    "seniority",
    "must_have_skills",
    "nice_to_have_skills",
    "city",
    "salary_min",
    "salary_max",
    "contract_type",
    "shift",
    "description_text",
    "summary",
    "source_filename",
]


class MemoryDB:
    def __init__(self):
        self.candidates = self._load_df(CANDIDATES_JSON, CANDIDATE_COLUMNS)
        self.jobs = self._load_df(JOBS_JSON, JOB_COLUMNS)
        self.interactions = []
        self.results = self._load_results()

    def _clean_df(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Keeps only required columns, adds missing columns, and replaces missing values
        without triggering pandas FutureWarning messages.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=columns)

        df = df.copy()

        for col in columns:
            if col not in df.columns:
                df[col] = ""

        df = df[columns]

        # Safer than fillna("") because it avoids pandas downcasting warnings.
        df = df.where(pd.notna(df), "")

        return df

    def _load_df(self, path, columns):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                df = pd.DataFrame(data)
                return self._clean_df(df, columns)
            except Exception:
                pass

        return pd.DataFrame(columns=columns)

    def _save_df(self, df, path):
        df = self._clean_df(df, list(df.columns))

        path.write_text(
            json.dumps(
                df.to_dict(orient="records"),
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    def _load_results(self):
        if RESULTS_JSON.exists():
            try:
                return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "overall_summary": "",
            "job_results": []
        }

    def _save_results(self):
        RESULTS_JSON.write_text(
            json.dumps(
                self.results,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    def candidates_list(self):
        if self.candidates.empty:
            return []

        df = self.candidates[
            ["id", "name", "specialization", "years_exp", "source_filename"]
        ].copy()

        df = df.where(pd.notna(df), "")

        return df.to_dict(orient="records")

    def jobs_list(self):
        if self.jobs.empty:
            return []

        df = self.jobs[
            ["id", "title", "specialization", "city", "source_filename"]
        ].copy()

        df = df.where(pd.notna(df), "")

        return df.to_dict(orient="records")

    def get_candidate(self, cid):
        rec = self.candidates[self.candidates["id"] == cid]
        return rec.iloc[0].to_dict() if len(rec) else None

    def get_job(self, jid):
        rec = self.jobs[self.jobs["id"] == jid]
        return rec.iloc[0].to_dict() if len(rec) else None

    def jobs_iter(self):
        for _, row in self.jobs.iterrows():
            yield row.to_dict()

    def add_candidate(self, cand: dict):
        new_row = self._clean_df(pd.DataFrame([cand]), CANDIDATE_COLUMNS)

        if self.candidates.empty:
            self.candidates = new_row
        else:
            self.candidates = pd.concat(
                [self.candidates, new_row],
                ignore_index=True
            )

        self.candidates = self._clean_df(self.candidates, CANDIDATE_COLUMNS)
        self._save_df(self.candidates, CANDIDATES_JSON)

    def add_job(self, job: dict):
        new_row = self._clean_df(pd.DataFrame([job]), JOB_COLUMNS)

        if self.jobs.empty:
            self.jobs = new_row
        else:
            self.jobs = pd.concat(
                [self.jobs, new_row],
                ignore_index=True
            )

        self.jobs = self._clean_df(self.jobs, JOB_COLUMNS)
        self._save_df(self.jobs, JOBS_JSON)

    def save_results(self, overall_summary: str, job_results: list):
        self.results = {
            "overall_summary": overall_summary,
            "job_results": job_results
        }

        self._save_results()

    def clear_candidates(self):
        self.candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
        self._save_df(self.candidates, CANDIDATES_JSON)

    def clear_jobs(self):
        self.jobs = pd.DataFrame(columns=JOB_COLUMNS)
        self._save_df(self.jobs, JOBS_JSON)

    def clear_results(self):
        self.results = {
            "overall_summary": "",
            "job_results": []
        }

        self._save_results()

    def clear_all(self):
        self.clear_candidates()
        self.clear_jobs()
        self.clear_results()
        self.interactions = []

    def log_interaction(self, event_dict):
        self.interactions.append(event_dict)