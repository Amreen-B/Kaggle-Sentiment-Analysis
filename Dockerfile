FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 -r requirements.txt
COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]