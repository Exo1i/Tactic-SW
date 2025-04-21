FROM python:3.11-slim

WORKDIR /app

COPY backend backend
COPY games games

RUN pip install --no-cache-dir fastapi uvicorn opencv-contrib-python numpy

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
