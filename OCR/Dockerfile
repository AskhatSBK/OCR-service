FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libmagic: content sniffing. LibreOffice Writer (headless): DOC/DOCX -> PDF.
# Fonts: metric-compatible MS Office fonts + DejaVu for Cyrillic/Greek.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libmagic1 \
        libreoffice-writer-nogui \
        fonts-dejavu-core \
        fonts-liberation2 \
        fonts-crosextra-carlito \
        fonts-crosextra-caladea \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock /app/
RUN pip install "poetry==2.1.3" \
    && poetry install --only main --no-root --no-ansi \
    && pip uninstall -y poetry \
    && rm -rf /root/.cache

COPY src /app/src
COPY entrypoint.sh /app/entrypoint.sh

# Non-root runtime user; LibreOffice needs a writable HOME.
RUN useradd --create-home --uid 10001 app \
    && chmod +x /app/entrypoint.sh \
    && chown -R app:app /app
USER app
ENV HOME=/home/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
