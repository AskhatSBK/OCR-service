"""Send test docs to the OCR service the way the main client does and score them against ground truth."""
import difflib
import json
import mimetypes
import re
import sys
import time
import types

import httpx

URL, DOCS = sys.argv[1].rstrip("/"), sys.argv[2]
TIMEOUT = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0  # client HTTP timeout


class DotsOCRError(Exception):
    pass


# --- Same request/response handling as the main client
def parse_file_sync(filename, data, content_type=None):
    files = {"file": (filename, data, content_type or "application/octet-stream")}
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{URL}/parse-file", files=files)
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPError as e:
        raise DotsOCRError(f"DotsOCR HTTP error: {e}") from e
    parts = []
    for r in payload.get("results") or []:
        md = (r.get("markdown_content") or "").strip() if isinstance(r, dict) else ""
        if md:
            parts.append(md)
    return "\n\n---\n\n".join(parts).strip()


def health_check_sync():
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{URL}/openapi.json")
        resp.raise_for_status()
        info = resp.json().get("info", {})
        return True, f"{info.get('title')} v{info.get('version')} reachable at {URL}"
    except Exception as e:
        return False, f"DotsOCR unreachable at {URL}: {e}"


dc = types.SimpleNamespace(parse_file_sync=parse_file_sync, health_check_sync=health_check_sync, DotsOCRError=DotsOCRError)

meta = json.load(open(f"{DOCS}/truth.json"))
print("health_check_sync:", dc.health_check_sync())


def norm(s):
    s = re.sub(r"[*#|`_>\-]+", " ", s)  # markdown/table syntax isn't an OCR error
    return re.sub(r"\s+", " ", s).strip().lower()


all_ok = True
for name, (kind, lines) in meta["truth"].items():
    data = open(f"{DOCS}/{name}", "rb").read()
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    t = time.time()
    try:
        text = dc.parse_file_sync(name, data, mime)
    except dc.DotsOCRError as e:
        print(f"\n### {name}: FAILED -> {e}")
        all_ok = False
        continue
    dt = time.time() - t
    fields = [f for k in kind.split("+") for f in meta["key_fields"][k]]
    found = [f for f in fields if norm(f) in norm(text)]
    sim = difflib.SequenceMatcher(None, norm(" ".join(lines)), norm(text)).ratio()
    missing = sorted(set(fields) - set(found))
    all_ok &= not missing
    print(f"\n### {name}: {dt:.1f}s  text-similarity={sim:.3f}  key-fields={len(found)}/{len(fields)}"
          + (f"  MISSING={missing}" if missing else ""))
    print(text[:1500])
print("\nALL KEY FIELDS FOUND" if all_ok else "\nSOME FIELDS MISSING")
sys.exit(0 if all_ok else 1)
