# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR Service - A containerized Document OCR microservice using FastAPI and vLLM with the dots.mocr vision-language model (`dots-studio/dots.mocr`, successor of `rednote-hilab/dots.ocr`; same architecture and prompts). Extracts text and layout information from documents (PDF, images, Office files).

## Commands

### Build and Run (Docker)
```bash
# Build and start all services (compose.yml)
docker compose up --build

# Start in background
docker compose up -d
```

### Tests (run inside the image; needs libmagic + LibreOffice)
```bash
cd OCR && docker build -t ocr-service:test .
docker run --rm --entrypoint bash -v "$PWD/tests:/app/tests:ro" ocr-service:test \
  -c "pip install -q --user pytest && python -m pytest -q -p no:cacheprovider tests"
```

### Local Development
```bash
cd OCR
poetry install
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Download Model Weights
```bash
# dots.mocr -> model/weights/DotsMOCR (default, mounted by compose.yml)
python dots.ocr/tools/download_model.py
# Roll back to dots.ocr: download it, then set OCR_MODEL_DIR=DotsOCR and restart dots-ocr
python dots.ocr/tools/download_model.py -n rednote-hilab/dots.ocr -d DotsOCR
```

### End-to-end smoke test (on the GPU host, stack running)
```bash
scripts/e2e/run.sh http://localhost:8001   # EN/RU/EL invoice + statement docs, client-style calls
```

### Test API
```bash
# Liveness / readiness (readiness checks vLLM serves the model)
curl http://localhost:8001/health
curl http://localhost:8001/ready

# Parse file
curl -X POST http://localhost:8001/parse-file -F "file=@document.pdf"
```

## Architecture

Two-container architecture:

1. **dots-ocr** (vLLM `v0.32.0-cu129`, needs NVIDIA driver >= 575, `--trust-remote-code` required) - Serves the OCR model via OpenAI-compatible API on port 8000 (internal)
2. **service** (FastAPI) - REST API on port 8001 (external), calls vLLM for inference

### Request Flow
```
Client → FastAPI /parse-file → DotsOCRParser → vLLM server → Response (JSON with markdown per page)
```

### Key Source Paths
- `OCR/src/main.py` - FastAPI app entry point
- `OCR/src/routers/v1.py` - API endpoint (`POST /parse-file`): limits, admission control, error mapping
- `OCR/src/security.py` - optional `X-API-Key` auth (off when `SERVICE_API_KEY` is empty)
- `OCR/src/dots_ocr/parser.py` - DotsOCRParser class (main parsing logic)
- `OCR/src/dots_ocr/model/inference.py` - vLLM inference calls
- `OCR/src/dots_ocr/utils/prompts.py` - Model prompts for different modes
- `OCR/src/settings.py` - Configuration from environment

### Supported Formats
PDF, JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP, HEIC, HEIF, DOC, DOCX (files without an extension are sniffed with libmagic)

### Error contract
400 empty/bad prompt_mode, 401 bad API key, 413 too large, 415 unsupported type, 422 corrupt document,
502 vLLM error, 503 busy (queue full, `Retry-After`), 504 vLLM timeout. Responses past `MAX_PAGES` are truncated (`truncated: true`).

### Client contract (keep stable)
The main client posts multipart `file` to `/parse-file`, joins `results[].markdown_content`, probes
`GET /openapi.json` (title/version) as its connectivity test, sends no auth header, uses a 120s HTTP timeout
and falls back to a local OCR engine on any non-2xx. Keep the path, field names and `/openapi.json` stable.

### Prompt Modes (PROMPT_MODE env var)
- `prompt_layout_all_en` - Full layout detection + OCR with bounding boxes
- `prompt_layout_only_en` - Layout detection only (no text)
- `prompt_ocr` - Simple text extraction to markdown

## Environment Variables

Configure in `OCR/.env` (see `.env.example` for all, with defaults in `OCR/src/settings.py`):
- `DOTS_OCR_HOST` / `DOTS_OCR_PORT` - vLLM server (default: `dots-ocr:8000`)
- `DOTS_OCR_BASE_URL` / `DOTS_OCR_API_KEY` - remote OpenAI-compatible endpoint (e.g. RunPod); overrides host/port
- `DOTS_OCR_MODEL` - Model name (default: `model`)
- `PROMPT_MODE` - OCR mode (default: `prompt_ocr`)
- `MAX_CONCURRENT_REQUESTS`, `QUEUE_TIMEOUT_SECONDS`, `PAGE_CONCURRENCY` - load control
- `MAX_UPLOAD_MB`, `MAX_PAGES`, `INFERENCE_TIMEOUT_SECONDS`, `LIBREOFFICE_TIMEOUT_SECONDS` - limits
- `SERVICE_API_KEY` - enables `X-API-Key` auth (leave empty unless every client sends the header)

Compose-level (shell or root `.env`): `VLLM_IMAGE` (pinned to `vllm/vllm-openai:v0.32.0-cu129`), `OCR_MODEL_DIR` (`DotsMOCR` default, `DotsOCR` to roll back), `OCR_GPU_DEVICE`, `VLLM_GPU_MEMORY_UTILIZATION`, `OCR_PUBLISH_PORT`.

## Dependencies

- Python 3.12+, Poetry for package management
- GPU required: NVIDIA drivers and runtime for vLLM
- System packages in container: LibreOffice Writer headless (DOC/DOCX conversion), libmagic, fonts; runs as non-root `app`
