FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
