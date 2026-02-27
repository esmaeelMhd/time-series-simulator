FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIMESIM_HOST=0.0.0.0 \
    TIMESIM_PORT=8000

WORKDIR /app

ARG REQUIREMENTS_FILE="requirements.txt"
ARG PIP_EXTRA_INDEX_URL=""

COPY requirements*.txt /app/
RUN pip install --upgrade pip && \
    if [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
      PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" pip install -r "/app/${REQUIREMENTS_FILE}"; \
    else \
      pip install -r "/app/${REQUIREMENTS_FILE}"; \
    fi

COPY . /app
RUN pip install -e .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["python", "scripts/docker_start_api.py"]
