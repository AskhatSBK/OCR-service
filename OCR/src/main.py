import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from loguru import logger

from src.routers.v1 import router as v1_router
from src.settings import SERVICE_VERSION, settings

logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL.upper(), backtrace=False, diagnose=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"OCR Service v{SERVICE_VERSION} starting: backend={settings.vllm_base_url} "
        f"model={settings.DOTS_OCR_MODEL} prompt_mode={settings.PROMPT_MODE} "
        f"max_concurrent={settings.MAX_CONCURRENT_REQUESTS} max_pages={settings.MAX_PAGES} "
        f"auth={'on' if settings.SERVICE_API_KEY else 'off'}"
    )
    yield
    logger.info("OCR Service stopped")


# NOTE: /openapi.json must stay enabled — clients probe it as a connectivity test.
app = FastAPI(title='OCR Service', version=SERVICE_VERSION, root_path=settings.ROOT_PATH, lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time.monotonic()
    with logger.contextualize(request_id=request_id):
        response = await call_next(request)
        if request.url.path not in ("/health", "/ping", "/ready"):
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} "
                f"({time.monotonic() - started:.2f}s)"
            )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", include_in_schema=False)
@app.get("/ping", include_in_schema=False)
async def health_check():
    """Liveness: the API process is up."""
    return Response(status_code=200)


@app.get("/ready", include_in_schema=False)
async def readiness_check():
    """Readiness: the vLLM backend answers and serves the configured model."""
    headers = {"Authorization": f"Bearer {settings.DOTS_OCR_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.vllm_base_url}/models", headers=headers)
        resp.raise_for_status()
        models = [m.get("id") for m in resp.json().get("data", [])]
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": f"vLLM unreachable: {e}"})
    if settings.DOTS_OCR_MODEL not in models:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": f"model '{settings.DOTS_OCR_MODEL}' not served; got {models}"},
        )
    return {"status": "ok", "model": settings.DOTS_OCR_MODEL}


app.include_router(v1_router)
