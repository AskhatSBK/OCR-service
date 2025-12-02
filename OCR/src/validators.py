from fastapi.exceptions import HTTPException
from fastapi import Header
from src.settings import settings

def validate_header(x_api_key: str = Header(...,  include_in_schema=False)):
    if x_api_key != settings.X_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key