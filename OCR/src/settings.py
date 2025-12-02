from os.path import join
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = join(BASE_DIR, 'files')


class Settings(BaseSettings):
    DOTS_OCR_PORT: int
    DOTS_OCR_HOST: str
    DOTS_OCR_MODEL: str
    PROMPT_MODE: str

    OCR_MODE: str #DotsOCR Docling

    ROOT_PATH: str

    model_config = SettingsConfigDict(env_file="./.env")


settings = Settings()
