FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUD_ARTIFACT_DIR=/app/artifacts/v1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 fraud

WORKDIR /app
COPY requirements.lock pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --requirement requirements.lock \
    && python -m pip install .

COPY artifacts/v1/ ./artifacts/v1/
COPY reports/locked/v1/ ./reports/locked/v1/
COPY app/ ./app/
RUN chown -R fraud:fraud /app

USER fraud
EXPOSE 8001 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health?deep=true', timeout=3)"

CMD ["uvicorn", "fraud_monitoring.api:app", "--host", "0.0.0.0", "--port", "8001"]
