# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR Service - A containerized Document OCR microservice using FastAPI and vLLM with the DotsOCR vision-language model. Extracts text and layout information from documents (PDF, images, Office files).

## Commands

### Build and Run (Docker)
```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d
```

### Local Development
```bash
cd OCR
poetry install
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Download Model Weights
```bash
cd dots.ocr
python tools/download_model.py --type huggingface --name rednote-hilab/dots.ocr
```

### Test API
```bash
# Health check
curl http://localhost:8001/health

# Parse file
curl -X POST http://localhost:8001/parse-file -F "file=@document.pdf"
```

## Architecture

Two-container architecture:

1. **dots-ocr** (vLLM server) - Serves DotsOCR model via OpenAI-compatible API on port 8000 (internal)
2. **service** (FastAPI) - REST API on port 8001 (external), calls vLLM for inference

### Request Flow
```
Client → FastAPI /parse-file → DotsOCRParser → vLLM server → Response (JSON with markdown per page)
```

### Key Source Paths
- `OCR/src/main.py` - FastAPI app entry point
- `OCR/src/routers/v1.py` - API endpoint (`POST /parse-file`)
- `OCR/src/dots_ocr/parser.py` - DotsOCRParser class (main parsing logic)
- `OCR/src/dots_ocr/model/inference.py` - vLLM inference calls
- `OCR/src/dots_ocr/utils/prompts.py` - Model prompts for different modes
- `OCR/src/settings.py` - Configuration from environment

### Supported Formats
PDF, JPG, JPEG, PNG, HEIC, HEIF, DOC, DOCX

### Prompt Modes (PROMPT_MODE env var)
- `prompt_layout_all_en` - Full layout detection + OCR with bounding boxes
- `prompt_layout_only_en` - Layout detection only (no text)
- `prompt_ocr` - Simple text extraction to markdown

## Environment Variables

Configure in `OCR/.env` (see `.env.example`):
- `DOTS_OCR_HOST` - vLLM server hostname (default: `dots-ocr`)
- `DOTS_OCR_PORT` - vLLM server port (default: `8000`)
- `DOTS_OCR_MODEL` - Model name (default: `model`)
- `PROMPT_MODE` - OCR mode (default: `prompt_ocr`)

## Dependencies

- Python 3.12+, Poetry for package management
- GPU required: NVIDIA drivers and runtime for vLLM
- System packages in container: LibreOffice (DOC/DOCX conversion), poppler-utils
