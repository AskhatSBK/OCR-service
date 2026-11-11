#!/bin/bash
set -e

echo "Starting OCR service..."

# Single worker on purpose: admission control (MAX_CONCURRENT_REQUESTS) is
# per-process, and the real bottleneck is the GPU behind vLLM, not this API.
# exec: uvicorn becomes PID 1 and receives SIGTERM for graceful shutdown.
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 75 \
    --timeout-graceful-shutdown "${GRACEFUL_SHUTDOWN_SECONDS:-120}" \
    --no-access-log \
    --no-server-header
