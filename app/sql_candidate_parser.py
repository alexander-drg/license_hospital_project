# sql_candidate_parser.py
import json
import pymysql
from typing import Dict, List, Any, Optional


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "NewPassword123!",
    "database": "dump_test",
    "cursorclass": pymysql.cursors.DictCursor,
}


IMPORTANT_CANDIDATE_TABLES = [
    "candidates",
    "candidate_addresses",
    "candidate_educations",
    "candidate_experiences",
    "candidate_domain_experiences",
    "candidate_spoken_language",
    "candidate_features",
    "candidate_tag_association",
]

IMPORTANT_NEED_TABLES = [
    "needs",
    "need_request",
    "need_comments",
    "need_schedule",
    "need_tag",
    "benefit_need",
    "collaboration_need",
    "experience_need",
    "companies",
]

LOOKUP_TABLES = [
    "tags",
    "cities",
    "countries",
    "education",
    "experiences",
    "spoken_languages",
    "categories",
    "subcategories",
]


def get_connection():
    return pymysql.connect(**DB_CONFIG)


def table_exists(conn, table_name: str) -> bool:
    query = """
        SELECT COUNT(*) AS count
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
        AND table_name = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (table_name,))
        result = cursor.fetchone()

    return result["count"] > 0


def get_table_columns(conn, table_name: str):
    query = """
        SELECT COLUMN_NAME
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
        AND table_name = %s
        ORDER BY ORDINAL_POSITION
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (table_name,))
        rows = cursor.fetchall()

    return [list(row.values())[0] for row in rows]


def fetch_rows_by_possible_candidate_id(
    conn,
    table_name: str,
    candidate_id: int,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Tries to fetch rows related to a candidate from tables that may use:
    candidate_id, candidateId, id_candidate, or candidate.
    """

    if not table_exists(conn, table_name):
        return []

    columns = get_table_columns(conn, table_name)

    possible_id_columns = [
        "candidate_id",
        "candidateId",
        "id_candidate",
        "candidate",
    ]

    id_column = None

    for col in possible_id_columns:
        if col in columns:
            id_column = col
            break

    if id_column is None:
        return []

    query = f"""
        SELECT *
        FROM `{table_name}`
        WHERE `{id_column}` = %s
        LIMIT %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (candidate_id, limit))
        return cursor.fetchall()


def fetch_rows_by_possible_need_id(
    conn,
    table_name: str,
    need_id: int,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Tries to fetch rows related to a job/need from tables that may use:
    need_id, needId, id_need, or need.
    """

    if not table_exists(conn, table_name):
        return []

    columns = get_table_columns(conn, table_name)

    possible_id_columns = [
        "need_id",
        "needId",
        "id_need",
        "need",
    ]

    id_column = None

    for col in possible_id_columns:
        if col in columns:
            id_column = col
            break

    if id_column is None:
        return []

    query = f"""
        SELECT *
        FROM `{table_name}`
        WHERE `{id_column}` = %s
        LIMIT %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (need_id, limit))
        return cursor.fetchall()


def fetch_one_by_id(conn, table_name: str, row_id: int) -> Optional[Dict[str, Any]]:
    if not table_exists(conn, table_name):
        return None

    columns = get_table_columns(conn, table_name)

    if "id" not in columns:
        return None

    query = f"""
        SELECT *
        FROM `{table_name}`
        WHERE id = %s
        LIMIT 1
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (row_id,))
        return cursor.fetchone()


def clean_value(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in ["none", "null", "nan", ""]:
        return ""

    return value


def row_to_readable_text(row: Dict[str, Any]) -> str:
    """
    Converts one SQL row into readable text, while skipping empty fields.
    """

    parts = []

    for key, value in row.items():
        cleaned = clean_value(value)

        if cleaned:
            parts.append(f"{key}: {cleaned}")

    return "; ".join(parts)


def rows_to_readable_block(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f"{title}: No data found."

    text = f"{title}:\n"

    for index, row in enumerate(rows, start=1):
        text += f"- {index}. {row_to_readable_text(row)}\n"

    return text.strip()


def build_candidate_profile(candidate_id: int) -> Dict[str, Any]:
    """
    Builds a structured candidate profile from the most useful SQL tables.
    This is the data you should send to Ollama, not the raw database.
    """

    conn = get_connection()

    try:
        candidate_main = fetch_one_by_id(conn, "candidates", candidate_id)

        if not candidate_main:
            return {
                "success": False,
                "message": f"No candidate found with id {candidate_id}",
                "candidate_id": candidate_id,
                "profile_text": "",
                "raw_data": {},
            }

        raw_data = {
            "candidates": [candidate_main]
        }

        for table in IMPORTANT_CANDIDATE_TABLES:
            if table == "candidates":
                continue

            rows = fetch_rows_by_possible_candidate_id(conn, table, candidate_id)
            raw_data[table] = rows

        profile_text = f"""
CANDIDATE PROFILE

Main candidate data:
{row_to_readable_text(candidate_main)}

{rows_to_readable_block("Candidate addresses", raw_data.get("candidate_addresses", []))}

{rows_to_readable_block("Candidate education", raw_data.get("candidate_educations", []))}

{rows_to_readable_block("Candidate experience", raw_data.get("candidate_experiences", []))}

{rows_to_readable_block("Candidate domain experience", raw_data.get("candidate_domain_experiences", []))}

{rows_to_readable_block("Candidate spoken languages", raw_data.get("candidate_spoken_language", []))}

{rows_to_readable_block("Candidate features / abilities", raw_data.get("candidate_features", []))}

{rows_to_readable_block("Candidate tags", raw_data.get("candidate_tag_association", []))}
""".strip()

        return {
            "success": True,
            "candidate_id": candidate_id,
            "profile_text": profile_text,
            "raw_data": raw_data,
        }

    finally:
        conn.close()


def build_need_profile(need_id: int) -> Dict[str, Any]:
    """
    Builds a structured job/need profile from the most useful SQL tables.
    """

    conn = get_connection()

    try:
        need_main = fetch_one_by_id(conn, "needs", need_id)

        if not need_main:
            return {
                "success": False,
                "message": f"No need found with id {need_id}",
                "need_id": need_id,
                "profile_text": "",
                "raw_data": {},
            }

        raw_data = {
            "needs": [need_main]
        }

        for table in IMPORTANT_NEED_TABLES:
            if table == "needs":
                continue

            rows = fetch_rows_by_possible_need_id(conn, table, need_id)
            raw_data[table] = rows

        profile_text = f"""
JOB / NEED PROFILE

Main need data:
{row_to_readable_text(need_main)}

{rows_to_readable_block("Need request", raw_data.get("need_request", []))}

{rows_to_readable_block("Need comments", raw_data.get("need_comments", []))}

{rows_to_readable_block("Need schedule", raw_data.get("need_schedule", []))}

{rows_to_readable_block("Need tags", raw_data.get("need_tag", []))}

{rows_to_readable_block("Required benefits", raw_data.get("benefit_need", []))}

{rows_to_readable_block("Required collaboration type", raw_data.get("collaboration_need", []))}

{rows_to_readable_block("Required experience", raw_data.get("experience_need", []))}
""".strip()

        return {
            "success": True,
            "need_id": need_id,
            "profile_text": profile_text,
            "raw_data": raw_data,
        }

    finally:
        conn.close()


def build_ollama_prompt_for_sql_match(candidate_id: int, need_id: int) -> Dict[str, Any]:
    """
    Final function you call from your app.
    It builds a clean prompt for Ollama using one candidate and one job/need.
    """

    candidate_profile = build_candidate_profile(candidate_id)
    need_profile = build_need_profile(need_id)

    if not candidate_profile["success"]:
        return candidate_profile

    if not need_profile["success"]:
        return need_profile

    prompt = f"""
You are an AI assistant used in a recruitment application.

Analyze the compatibility between the following candidate and the following job/need.

Use only the information provided below.
Do not invent missing details.
If data is missing or anonymized, mention that clearly.

Candidate:
{candidate_profile["profile_text"]}

Job / Need:
{need_profile["profile_text"]}

Return the answer in this structure:

1. Compatibility score from 0 to 100
2. Candidate strengths
3. Missing or weak requirements
4. Relevant experience
5. Final recommendation:
   - Strong match
   - Good match
   - Medium match
   - Weak match
6. Short explanation suitable for a recruitment dashboard
""".strip()

    return {
        "success": True,
        "candidate_id": candidate_id,
        "need_id": need_id,
        "prompt": prompt,
        "candidate_profile": candidate_profile["profile_text"],
        "need_profile": need_profile["profile_text"],
    }


def list_candidates(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Used for dropdown/list in your SQL page.
    """

    conn = get_connection()

    try:
        if not table_exists(conn, "candidates"):
            return []

        query = """
            SELECT *
            FROM candidates
            LIMIT %s
        """

        with conn.cursor() as cursor:
            cursor.execute(query, (limit,))
            return cursor.fetchall()

    finally:
        conn.close()


def list_needs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Used for dropdown/list in your SQL page.
    """

    conn = get_connection()

    try:
        if not table_exists(conn, "needs"):
            return []

        query = """
            SELECT *
            FROM needs
            LIMIT %s
        """

        with conn.cursor() as cursor:
            cursor.execute(query, (limit,))
            return cursor.fetchall()

    finally:
        conn.close()


if __name__ == "__main__":
    # Test example.
    # Change these IDs after you inspect your database.
    candidate_id = 1
    need_id = 1

    result = build_ollama_prompt_for_sql_match(candidate_id, need_id)

    if result["success"]:
        print(result["prompt"])
    else:
        print(result["message"])
        
def list_database_tables() -> List[str]:
    conn = get_connection()
    try:
        query = "SHOW TABLES"
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        tables = []
        for row in rows:
            tables.append(list(row.values())[0])

        return sorted(tables)
    finally:
        conn.close()


def fetch_table_sample(table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        if not table_exists(conn, table_name):
            return []

        query = f"SELECT * FROM `{table_name}` LIMIT %s"

        with conn.cursor() as cursor:
            cursor.execute(query, (limit,))
            return cursor.fetchall()
    finally:
        conn.close()


def build_multi_table_profile(table_names: List[str], limit_per_table: int = 3) -> Dict[str, Any]:
    sections = {}
    readable_parts = []

    for table in table_names:
        rows = fetch_table_sample(table, limit=limit_per_table)
        sections[table] = rows

        readable_parts.append(
            rows_to_readable_block(f"TABLE: {table}", rows)
        )

    profile_text = "\n\n".join(readable_parts)

    return {
        "success": True,
        "tables": table_names,
        "profile_text": profile_text,
        "raw_data": sections,
    }        
    
def build_multi_table_schema_profile(table_names: List[str], limit_per_table: int = 1) -> Dict[str, Any]:
    conn = get_connection()

    try:
        readable_parts = []

        for table in table_names:
            if not table_exists(conn, table):
                continue

            columns = get_table_columns(conn, table)

            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) AS row_count FROM `{table}`")
                count_row = cursor.fetchone()
                row_count = list(count_row.values())[0] if count_row else 0

            sample_rows = fetch_table_sample(table, limit=limit_per_table)

            compact_samples = []
            for row in sample_rows:
                compact_row = {}

                for key, value in row.items():
                    cleaned = clean_value(value)

                    if not cleaned:
                        continue

                    if len(cleaned) > 80:
                        cleaned = cleaned[:80] + "..."

                    compact_row[key] = cleaned

                compact_samples.append(compact_row)

            readable_parts.append(
                f"""
TABLE: {table}
Columns: {", ".join(columns)}
Row count: {row_count}
Sample rows:
{json.dumps(compact_samples, ensure_ascii=False, indent=2)}
""".strip()
            )

        profile_text = "\n\n".join(readable_parts)

        return {
            "success": True,
            "tables": table_names,
            "profile_text": profile_text,
        }

    finally:
        conn.close()
        
def build_table_ai_summary(table_names: List[str]) -> Dict[str, Any]:
    conn = get_connection()

    try:
        summaries = []

        for table in table_names:
            if not table_exists(conn, table):
                continue

            columns = get_table_columns(conn, table)

            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) AS row_count FROM `{table}`")
                count_row = cursor.fetchone()
                row_count = list(count_row.values())[0] if count_row else 0

            link_columns = [
                c for c in columns
                if c.endswith("_id") or c in ["candidate_id", "need_id", "job_id", "company_id", "city_id", "category_id"]
            ]

            summaries.append({
                "table": table,
                "row_count": row_count,
                "columns": columns[:25],
                "link_columns": link_columns,
            })

        profile_text = json.dumps(summaries, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "tables": table_names,
            "profile_text": profile_text,
        }

    finally:
        conn.close()        