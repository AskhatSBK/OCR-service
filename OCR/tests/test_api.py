import asyncio
import io
import shutil

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.dots_ocr import parser as parser_module
from src.dots_ocr.model.inference import InferenceError, InferenceTimeout
from src.main import app
from src.routers import v1
from src.settings import SERVICE_VERSION, settings


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake_vllm(monkeypatch):
    """Replace the vLLM call; answers 'TEXT <n>' where n counts calls."""
    calls = []

    def fake(image, prompt, **kwargs):
        calls.append(kwargs)
        return f"TEXT {len(calls)}"

    monkeypatch.setattr(parser_module, "inference_with_vllm", fake)
    return calls


def _image_bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), "white").save(buf, format=fmt)
    return buf.getvalue()


def _pdf_bytes(pages: int) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i}")
    data = doc.tobytes()
    doc.close()
    return data


def client_join_results(payload: dict) -> str:
    """How the main client turns a response into one string."""
    results = payload.get("results") or []
    assert isinstance(results, list)
    parts = []
    for r in results:
        if not isinstance(r, dict):
            continue
        md = (r.get("markdown_content") or "").strip()
        if md:
            parts.append(md)
    return "\n\n---\n\n".join(parts).strip()


# --- health / probes -------------------------------------------------------

def test_health_and_ping(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ping").status_code == 200


def test_openapi_probe_used_by_clients(client):
    info = client.get("/openapi.json").json()["info"]
    assert info["title"] == "OCR Service"
    assert info["version"] == SERVICE_VERSION


def test_ready_fails_when_vllm_down(client, monkeypatch):
    monkeypatch.setattr(settings, "DOTS_OCR_BASE_URL", "http://127.0.0.1:9/v1")
    assert client.get("/ready").status_code == 503


# --- happy paths -----------------------------------------------------------

def test_png_matches_client_contract(client, fake_vllm):
    resp = client.post("/parse-file", files={"file": ("scan.png", _image_bytes("PNG"), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "scan.png"
    assert body["results"] == [{"page_no": 0, "markdown_content": "TEXT 1"}]
    assert body["truncated"] is False
    assert client_join_results(body) == "TEXT 1"
    assert "X-Request-ID" in resp.headers


@pytest.mark.parametrize("fmt,name,mime", [
    ("WEBP", "a.webp", "image/webp"),
    ("GIF", "a.gif", "image/gif"),
    ("BMP", "a.bmp", "image/bmp"),
    ("TIFF", "a.tiff", "image/tiff"),
    ("JPEG", "A.JPG", "image/jpeg"),
])
def test_image_formats_clients_send(client, fake_vllm, fmt, name, mime):
    resp = client.post("/parse-file", files={"file": (name, _image_bytes(fmt), mime)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"][0]["markdown_content"] == "TEXT 1"


def test_pdf_pages_in_order(client, fake_vllm):
    resp = client.post("/parse-file", files={"file": ("doc.pdf", _pdf_bytes(5), "application/pdf")})
    assert resp.status_code == 200
    body = resp.json()
    assert [r["page_no"] for r in body["results"]] == [0, 1, 2, 3, 4]
    assert body["total_pages"] == 5 and body["processed_pages"] == 5


def test_pdf_truncated_at_max_pages(client, fake_vllm, monkeypatch):
    monkeypatch.setattr(settings, "MAX_PAGES", 2)
    body = client.post("/parse-file", files={"file": ("doc.pdf", _pdf_bytes(4), "application/pdf")}).json()
    assert body["processed_pages"] == 2 and body["total_pages"] == 4 and body["truncated"] is True
    assert len(fake_vllm) == 2


def test_extensionless_upload_is_sniffed(client, fake_vllm):
    resp = client.post("/parse-file", files={"file": ("image", _image_bytes("PNG"), "application/octet-stream")})
    assert resp.status_code == 200


def test_inline_base64_images_stripped(client, monkeypatch):
    monkeypatch.setattr(parser_module, "inference_with_vllm",
                        lambda image, prompt, **kw: "a ![](data:image/png;base64,AAAA) b")
    body = client.post("/parse-file", files={"file": ("x.png", _image_bytes("PNG"), "image/png")}).json()
    assert "base64" not in body["results"][0]["markdown_content"]


@pytest.mark.skipif(not (shutil.which("soffice") or shutil.which("libreoffice")), reason="LibreOffice not installed")
def test_docx(client, fake_vllm):
    import docx
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph("hello")
    d.save(buf)
    resp = client.post("/parse-file", files={"file": ("a.docx", buf.getvalue(), "application/octet-stream")})
    assert resp.status_code == 200, resp.text


# --- failure modes (clients fall back to local OCR on any non-2xx) ---------

def test_unsupported_format(client):
    resp = client.post("/parse-file", files={"file": ("a.txt", b"just text", "text/plain")})
    assert resp.status_code == 415


def test_empty_file(client):
    assert client.post("/parse-file", files={"file": ("a.png", b"", "image/png")}).status_code == 400


def test_too_large(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    resp = client.post("/parse-file", files={"file": ("a.pdf", b"0" * (1024 * 1024 + 1), "application/pdf")})
    assert resp.status_code == 413


def test_corrupt_pdf(client, fake_vllm):
    resp = client.post("/parse-file", files={"file": ("a.pdf", b"%PDF-1.4 garbage", "application/pdf")})
    assert resp.status_code == 422


def test_backend_error_is_502(client, monkeypatch):
    def boom(*a, **kw):
        raise InferenceError("vLLM request failed: connection refused")
    monkeypatch.setattr(parser_module, "inference_with_vllm", boom)
    assert client.post("/parse-file", files={"file": ("a.png", _image_bytes("PNG"), "image/png")}).status_code == 502


def test_backend_timeout_is_504(client, monkeypatch):
    def slow(*a, **kw):
        raise InferenceTimeout("timed out")
    monkeypatch.setattr(parser_module, "inference_with_vllm", slow)
    assert client.post("/parse-file", files={"file": ("a.png", _image_bytes("PNG"), "image/png")}).status_code == 504


def test_busy_is_503(client, fake_vllm, monkeypatch):
    monkeypatch.setattr(v1, "_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(settings, "QUEUE_TIMEOUT_SECONDS", 0.01)
    resp = client.post("/parse-file", files={"file": ("a.png", _image_bytes("PNG"), "image/png")})
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "30"


def test_bad_prompt_mode(client):
    resp = client.post("/parse-file", data={"prompt_mode": "nope"},
                       files={"file": ("a.png", _image_bytes("PNG"), "image/png")})
    assert resp.status_code == 400


# --- auth ------------------------------------------------------------------

def test_api_key_disabled_by_default(client, fake_vllm):
    assert settings.SERVICE_API_KEY == ""
    assert client.post("/parse-file", files={"file": ("a.png", _image_bytes("PNG"), "image/png")}).status_code == 200


def test_api_key_enforced_when_set(client, fake_vllm, monkeypatch):
    monkeypatch.setattr(settings, "SERVICE_API_KEY", "s3cret")
    files = {"file": ("a.png", _image_bytes("PNG"), "image/png")}
    assert client.post("/parse-file", files=files).status_code == 401
    assert client.post("/parse-file", files=files, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_request_deadline_stops_sending_pages(client, fake_vllm, monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_DEADLINE_SECONDS", 0.5)
    resp = client.post("/parse-file", files={"file": ("doc.pdf", _pdf_bytes(3), "application/pdf")})
    assert resp.status_code == 504
    assert fake_vllm == []


def test_defaults_match_client():
    assert settings.MAX_PAGES == 20  # client-side page cap
    assert settings.REQUEST_DEADLINE_SECONDS < 120  # client HTTP timeout
    assert settings.MAX_UPLOAD_MB >= 40  # largest client upload
