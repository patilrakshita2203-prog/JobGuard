# One image for both services (API and dashboard); docker-compose picks the command.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -c "import nltk; [nltk.download(p, download_dir='/usr/local/share/nltk_data') for p in ('stopwords','punkt','punkt_tab','wordnet')]"

COPY . .

# Run as a non-root user; /app/data holds the SQLite prediction log
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app/data
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Default: the FastAPI backend. The dashboard overrides this command (see docker-compose.yml).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
