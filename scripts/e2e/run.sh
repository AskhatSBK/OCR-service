#!/usr/bin/env bash
# End-to-end smoke test against a running stack (run on the GPU host after `docker compose up -d`).
#   scripts/e2e/run.sh [OCR_URL]        default http://localhost:8001
# Generates EN/RU/EL invoice + bank-statement documents (digital PDF, scanned PDF,
# WebP photo, PNG) with known text, sends them like the main client does, checks key fields.
set -euo pipefail
URL="${1:-http://localhost:8001}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${OCR_IMAGE:-ocr-service:latest}"

echo "== /ready"; curl -fsS "$URL/ready"; echo
docker run --rm --network host --entrypoint bash -v "$HERE:/e2e:ro" "$IMAGE" -c \
  "mkdir -p /tmp/docs && python /e2e/make_docs.py /tmp/docs && python /e2e/eval_docs.py '$URL' /tmp/docs"
