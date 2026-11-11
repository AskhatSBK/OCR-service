import asyncio
import os
import re
import tempfile
import time

import magic
import pymupdf
from PIL import UnidentifiedImageError
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from src.dots_ocr.model.inference import InferenceError, InferenceTimeout
from src.dots_ocr.parser import DotsOCRParser, UnsupportedDocumentError
from src.dots_ocr.preprocess import PreprocessError
from src.dots_ocr.utils.prompts import dict_promptmode_to_prompt
from src.security import require_api_key
from src.settings import settings

router = APIRouter(tags=["v1"])

ALLOWED_EXTENSIONS = {
    '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp',
    '.heic', '.heif', '.doc', '.docx',
}
# Used when the filename has no/unknown extension: trust the bytes instead.
_MIME_TO_EXT = {
    'application/pdf': '.pdf',
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/bmp': '.bmp',
    'image/x-ms-bmp': '.bmp',
    'image/gif': '.gif',
    'image/tiff': '.tiff',
    'image/webp': '.webp',
    'image/heic': '.heic',
    'image/heif': '.heif',
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
}
_INLINE_IMAGE_RE = re.compile(r"data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+")
_READ_CHUNK = 1024 * 1024

_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)


def _release_slot(task: asyncio.Future) -> None:
    _semaphore.release()
    if not task.cancelled():
        task.exception()  # mark as retrieved; the awaiting request handles it


def strip_inline_images(content: str) -> str:
    return _INLINE_IMAGE_RE.sub("", content)


def _resolve_extension(filename: str, head: bytes) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in ALLOWED_EXTENSIONS:
        return ext
    try:
        mime = magic.from_buffer(head, mime=True)
    except Exception:
        mime = ""
    sniffed = _MIME_TO_EXT.get(mime)
    if sniffed:
        return sniffed
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file format. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
    )


async def _read_limited(file: UploadFile) -> bytes:
    limit = settings.max_upload_bytes
    chunks, size = [], 0
    while chunk := await file.read(_READ_CHUNK):
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_MB} MB limit")
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    return b"".join(chunks)


def _build_parser(deadline: float) -> DotsOCRParser:
    return DotsOCRParser(
        base_url=settings.vllm_base_url,
        api_key=settings.DOTS_OCR_API_KEY,
        model_name=settings.DOTS_OCR_MODEL,
        max_completion_tokens=settings.MAX_COMPLETION_TOKENS,
        num_thread=settings.PAGE_CONCURRENCY,
        dpi=settings.PDF_DPI,
        timeout=settings.INFERENCE_TIMEOUT_SECONDS,
        max_retries=settings.INFERENCE_MAX_RETRIES,
        office_timeout=settings.LIBREOFFICE_TIMEOUT_SECONDS,
        deadline=deadline,
    )


def _process(content: bytes, ext: str, prompt_mode: str, deadline: float) -> tuple[list[dict], int]:
    """Blocking OCR pipeline; runs in a worker thread."""
    with tempfile.TemporaryDirectory(prefix="ocr_req_") as workdir:
        input_dir = os.path.join(workdir, "input")
        output_dir = os.path.join(workdir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        input_path = os.path.join(input_dir, f"document{ext}")
        with open(input_path, "wb") as f:
            f.write(content)

        results, total_pages = _build_parser(deadline).parse_file(
            input_path=input_path,
            output_dir=output_dir,
            fitz_preprocess=True,
            prompt_mode=prompt_mode,
            max_pages=settings.MAX_PAGES,
        )

        pages = []
        for result in results:
            md = result.get("md_content")
            if md is None and result.get("md_content_path") and os.path.exists(result["md_content_path"]):
                with open(result["md_content_path"], "r", encoding="utf-8") as f:
                    md = f.read()
            page = {"page_no": result.get("page_no", 0)}
            if md is not None:
                page["markdown_content"] = strip_inline_images(md)
            pages.append(page)
        return pages, total_pages


@router.post("/parse-file", dependencies=[Depends(require_api_key)])
async def parse_file(
    file: UploadFile = File(...),
    prompt_mode: str | None = Form(None),
):
    """
    Parse a document with DotsOCR.

    Supported formats: PDF, JPG/JPEG, PNG, BMP, GIF, TIFF, WEBP, HEIC/HEIF, DOC, DOCX.

    Response: `{"filename", "results": [{"page_no", "markdown_content"}], "total_pages",
    "processed_pages", "truncated"}`.
    """
    prompt_mode = prompt_mode or settings.PROMPT_MODE
    if prompt_mode not in dict_promptmode_to_prompt or prompt_mode == "prompt_grounding_ocr":
        raise HTTPException(status_code=400, detail=f"Unsupported prompt_mode '{prompt_mode}'")

    content = await _read_limited(file)
    ext = _resolve_extension(file.filename, content[:8192])

    received = time.monotonic()
    deadline = received + settings.REQUEST_DEADLINE_SECONDS
    try:
        await asyncio.wait_for(_semaphore.acquire(), timeout=settings.QUEUE_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(f"Rejecting {file.filename!r}: service busy")
        raise HTTPException(status_code=503, detail="OCR service is busy, retry later",
                            headers={"Retry-After": "30"})

    started = time.monotonic()
    # The slot is released when the worker thread finishes, not when this
    # coroutine exits: if the client disconnects, the thread keeps running and
    # must keep counting against MAX_CONCURRENT_REQUESTS.
    task = asyncio.ensure_future(run_in_threadpool(_process, content, ext, prompt_mode, deadline))
    task.add_done_callback(_release_slot)
    try:
        pages, total_pages = await asyncio.shield(task)
    except (UnsupportedDocumentError, PreprocessError, pymupdf.FileDataError, UnidentifiedImageError) as e:
        logger.warning(f"Cannot process {file.filename!r}: {e}")
        raise HTTPException(status_code=422, detail=f"Cannot process document: {e}")
    except InferenceTimeout as e:
        logger.error(f"OCR backend timeout for {file.filename!r}: {e}")
        raise HTTPException(status_code=504, detail=str(e))
    except InferenceError as e:
        logger.error(f"OCR backend error for {file.filename!r}: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        # fitz / PIL raise a zoo of exception types for corrupt input.
        logger.exception(f"Failed to parse {file.filename!r}")
        raise HTTPException(status_code=500, detail=f"Error parsing file: {e}")

    logger.info(
        f"Parsed {file.filename!r} ({len(content)} bytes, {len(pages)}/{total_pages} pages) "
        f"in {time.monotonic() - started:.2f}s"
    )
    return {
        "filename": file.filename,
        "results": pages,
        "total_pages": total_pages,
        "processed_pages": len(pages),
        "truncated": len(pages) < total_pages,
    }
