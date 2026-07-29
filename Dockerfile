FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY app ./app
COPY Makefile ./

# Build a model at image build time so the container is demo-ready on first run.
RUN python -m predmaint.cli demo-data && python -m predmaint.cli train --no-cv

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
