from os.path import join
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = join(BASE_DIR, 'files')

SERVICE_VERSION = "1.1.0"


class Settings(BaseSettings):
    # --- vLLM / DotsOCR backend -------------------------------------------
    DOTS_OCR_HOST: str = "dots-ocr"
    DOTS_OCR_PORT: int = 8000
    # Full OpenAI-compatible base URL (e.g. RunPod serverless). When set it
    # overrides DOTS_OCR_HOST / DOTS_OCR_PORT.
    DOTS_OCR_BASE_URL: str = ""
    DOTS_OCR_API_KEY: str = "0"
    DOTS_OCR_MODEL: str = "model"
    PROMPT_MODE: str = "prompt_ocr"

    OCR_MODE: str = "DOTS"
    ROOT_PATH: str = ""

    # Defaults below are matched to the main client: 120s HTTP timeout, 20-page cap,
    # uploads up to 40 MB. Change both sides together.

    # --- Inference tuning -------------------------------------------------
    INFERENCE_TIMEOUT_SECONDS: float = Field(60.0, gt=0)
    INFERENCE_MAX_RETRIES: int = Field(1, ge=0)
    # Whole-request budget (queue wait + all pages), kept under the caller's
    # HTTP timeout so no GPU work is spent on answers nobody will read.
    REQUEST_DEADLINE_SECONDS: float = Field(110.0, gt=0)
    MAX_COMPLETION_TOKENS: int = Field(16384, gt=0)
    PDF_DPI: int = Field(200, ge=72, le=400)
    # Pages of a single document sent to vLLM in parallel.
    PAGE_CONCURRENCY: int = Field(8, ge=1)

    # --- Admission control / limits --------------------------------------
    # Documents processed at the same time. Extra requests wait in a queue
    # for up to QUEUE_TIMEOUT_SECONDS, then get 503 so callers can fall back.
    MAX_CONCURRENT_REQUESTS: int = Field(4, ge=1)
    QUEUE_TIMEOUT_SECONDS: float = Field(20.0, ge=0)
    MAX_UPLOAD_MB: int = Field(50, ge=1)
    # Pages OCR'd per document; the rest are skipped and the response is
    # flagged `truncated: true`.
    MAX_PAGES: int = Field(20, ge=1)
    LIBREOFFICE_TIMEOUT_SECONDS: int = Field(120, ge=10)

    # --- Security ---------------------------------------------------------
    # When non-empty, /parse-file requires header `X-API-Key: <value>`.
    SERVICE_API_KEY: str = ""

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file="./.env", extra="ignore")

    @property
    def vllm_base_url(self) -> str:
        if self.DOTS_OCR_BASE_URL:
            return self.DOTS_OCR_BASE_URL.rstrip("/")
        return f"http://{self.DOTS_OCR_HOST}:{self.DOTS_OCR_PORT}/v1"

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()
