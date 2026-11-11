import secrets

from fastapi import Header, HTTPException

from src.settings import settings


def require_api_key(x_api_key: str | None = Header(None, include_in_schema=False)) -> None:
    """Optional shared-secret auth. Disabled while SERVICE_API_KEY is empty."""
    expected = settings.SERVICE_API_KEY
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
