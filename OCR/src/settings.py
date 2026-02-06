from os.path import join
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = join(BASE_DIR, 'files')


class Settings(BaseSettings):
    DOTS_OCR_PORT: int = 8000
    DOTS_OCR_HOST: str = "dots-ocr"
    DOTS_OCR_BASE_URL: str = ""
    DOTS_OCR_API_KEY: str = "0"
    DOTS_OCR_MODEL: str = "model"
    PROMPT_MODE: str = "prompt_ocr"

    OCR_MODE: str = "DOTS"

    ROOT_PATH: str = "/"

    model_config = SettingsConfigDict(env_file="./.env")


settings = Settings()
