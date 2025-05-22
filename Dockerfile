FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK 리소스 사전 설치 (3.7에서는 punkt만 필요)
RUN python -m nltk.downloader --quiet punkt

COPY src ./src

# ── FastAPI 서버로 기동 ──
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8125"]
