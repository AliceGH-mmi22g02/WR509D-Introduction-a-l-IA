FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir transformers torch fastapi pydantic uvicorn

RUN pip install --no-cache-dir redis

COPY app.py app.py

CMD ["python", "app.py"]
