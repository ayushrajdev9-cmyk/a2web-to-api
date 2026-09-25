FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY a2web2api ./a2web2api
COPY config.example.json ./config.example.json

ENV PYTHONUNBUFFERED=1
EXPOSE 8081

# mount your config.json (with cookie paths) into /app/config.json
CMD ["python", "-m", "a2web2api", "--config", "/app/config.json"]