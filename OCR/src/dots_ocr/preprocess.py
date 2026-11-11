import argparse
import atexit
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

# Optional content sniffer (libmagic). Falls back to mimetypes if unavailable.
try:
    import magic  # python-magic
    _HAVE_MAGIC = True
except Exception:
    _HAVE_MAGIC = False

# HEIC/HEIF support for Pillow
try:
    from PIL import Image, ImageOps
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HAVE_PIL_HEIF = True
except Exception:
    _HAVE_PIL_HEIF = False



class PreprocessError(RuntimeError):
    """Input could not be normalised (corrupt file, conversion failure)."""


def _mime_by_content(path: str) -> Optional[str]:
    if _HAVE_MAGIC:
        try:
            m = magic.Magic(mime=True)
            return m.from_file(path)
        except Exception:
            return None
    return mimetypes.guess_type(path)[0]


def _find_soffice_cmd() -> str:
    # LibreOffice installs both `soffice` and `libreoffice` on many systems; prefer whatever exists.
    for candidate in ("soffice", "libreoffice"):
        exe = shutil.which(candidate)
        if exe:
            return exe
    raise RuntimeError("LibreOffice not found (need 'soffice' or 'libreoffice' on PATH).")


def _ensure_workdir(workdir: Optional[str]) -> str:
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        return workdir
    return tempfile.mkdtemp(prefix="dots_preproc_")


def _lowercase_ext_alias(src_path: str, workdir: str) -> str:
    base = os.path.basename(src_path)
    root, ext = os.path.splitext(base)
    # If no extension or already lowercase, just return original path
    if not ext or ext == ext.lower():
        return src_path
    lower_name = f"{root}{ext.lower()}"
    dst_path = os.path.join(workdir, lower_name)
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return src_path
    # Copy (not symlink) for cross-platform safety
    shutil.copy2(src_path, dst_path)
    logger.info(f"Created lowercase-extension alias: {dst_path}")
    return dst_path


def convert_office_to_pdf(input_path: str, outdir: str, timeout_sec: int = 120) -> str:
    soffice = _find_soffice_cmd()
    # A private LibreOffice profile per conversion: concurrent soffice
    # processes sharing the default profile lock each other out and either
    # fail silently or hang.
    profile_dir = tempfile.mkdtemp(prefix="lo_profile_", dir=outdir)
    cmd = [
        soffice,
        f"-env:UserInstallation={Path(profile_dir).as_uri()}",
        "--headless",
        "--norestore",
        "--nologo",
        "--convert-to", "pdf",
        "--outdir", outdir,
        input_path,
    ]
    logger.info("Running: {}", " ".join(cmd))
    # New session so the whole soffice process tree can be killed on timeout.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise PreprocessError(f"LibreOffice conversion timed out after {timeout_sec}s")
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)

    if proc.returncode != 0:
        raise PreprocessError(
            f"LibreOffice conversion failed (rc={proc.returncode}): {stderr.strip() or stdout.strip()}"
        )
    base = os.path.splitext(os.path.basename(input_path))[0]
    pdf_path = os.path.join(outdir, f"{base}.pdf")
    if not os.path.exists(pdf_path):
        raise PreprocessError("LibreOffice could not convert the document (corrupt or unsupported file).")
    logger.info(f"DOC/DOCX → PDF: {pdf_path}")
    return pdf_path


def convert_heif_to_png(input_path: str, outdir: str) -> str:
    if not _HAVE_PIL_HEIF:
        raise PreprocessError("pillow-heif/Pillow not available; cannot convert HEIC/HEIF.")
    with Image.open(input_path) as im:
        # Fix common EXIF orientation issues and strip the orientation tag
        im = ImageOps.exif_transpose(im)
        # Ensure 8-bit sRGB-like RGB; drop alpha for OCR stability if present
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        if im.mode == "RGBA":
            # Composite onto white background to remove alpha
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            im = bg
        base = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(outdir, f"{base}.png")
        im.save(out_path, format="PNG", compress_level=6)
        logger.info(f"HEIC/HEIF → PNG: {out_path}")
        return out_path


def preprocess_path(
    input_path: str,
    workdir: Optional[str] = None,
    cleanup: bool = False,
    office_timeout_sec: int = 120,
) -> str:
    """
    Normalize an input path into a dots.ocr-friendly file:
      - lowercase extension alias for case-sensitive checks
      - DOC/DOCX -> PDF (LibreOffice headless)
      - HEIC/HEIF -> PNG (pillow-heif)
      - otherwise pass-through (or the lowercase alias)
    Returns the path to the normalized file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    tmpdir = _ensure_workdir(workdir)

    if cleanup and not workdir:
        # Clean temp workdir at process exit if we created it.
        atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    # Always create a lowercase alias when extension contains uppercase letters
    lower_alias = _lowercase_ext_alias(input_path, tmpdir)
    use_path = lower_alias

    # Determine type by content first
    mime = _mime_by_content(use_path) or ""
    ext = os.path.splitext(use_path)[1].lower()

    # Fast-path for PDF
    if mime == "application/pdf" or ext == ".pdf":
        return use_path

    # Office docs
    if mime in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    } or ext in {".docx", ".doc"}:
        return convert_office_to_pdf(use_path, tmpdir, timeout_sec=office_timeout_sec)

    # HEIC/HEIF
    if mime in {"image/heif", "image/heic", "image/heif-sequence", "image/heic-sequence"} or ext in {".heic", ".heif"}:
        return convert_heif_to_png(use_path, tmpdir)

    # Supported images: ensure lowercase extension only
    if ext in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".gif"} or mime.startswith("image/"):
        return use_path

    # Unknown type: last resort—just return the lowercase alias
    logger.warning(f"Unrecognized type (MIME='{mime}', ext='{ext}'); passing through: {use_path}")
    return use_path


def main():
    parser = argparse.ArgumentParser(description="Normalize path for dots.ocr (DOC/DOCX→PDF, HEIC→PNG, fix extension case).")
    parser.add_argument("input_path", type=str, help="Input file path")
    parser.add_argument("--workdir", type=str, default=None, help="Work directory for converted files (default: temp dir)")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not auto-clean temp dir on exit (default: auto-clean if we create it)")
    args = parser.parse_args()

    try:
        normalized = preprocess_path(args.input_path, workdir=args.workdir, cleanup=(not args.no_cleanup))
        print(normalized, end="")
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
