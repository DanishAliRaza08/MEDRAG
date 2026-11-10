FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MEDRAG_DATA_DIR=/data MEDRAG_KB_DIR=/app/data/knowledge_base
# Pass --build-arg EXTRAS=real to include the GPU/real-mode model dependencies.
ARG EXTRAS=""

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY medrag ./medrag
COPY data ./data
RUN pip install ".${EXTRAS:+[$EXTRAS]}"

RUN useradd --create-home medrag && mkdir -p /data && chown medrag /data
USER medrag
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["medrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
