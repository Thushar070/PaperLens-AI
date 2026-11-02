FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY backend/requirements-phase9.txt /app/requirements-phase9.txt
RUN pip install --no-cache-dir -r /app/requirements-phase9.txt
COPY backend/app /app/app
COPY ml /app/ml

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
