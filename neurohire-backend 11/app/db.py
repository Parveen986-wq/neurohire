"""Optional persistence to Supabase/PostgreSQL. Skipped silently if env vars are missing."""
import os

_client = None


def _get():
    global _client
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    if _client is None:
        from supabase import create_client
        _client = create_client(url, key)
    return _client


def save_analysis(filename: str | None, result: dict):
    c = _get()
    if not c:
        return
    try:
        c.table("analyses").insert({
            "filename": filename,
            "candidate_name": result["candidate"]["name"],
            "ats_score": result["scores"]["ats_score"],
            "top_role": result["predicted_roles"][0]["role"],
            "result": result,
        }).execute()
    except Exception as e:  # never fail the request because of storage
        print("Supabase insert failed:", e)
