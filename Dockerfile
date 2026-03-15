FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying source so this layer is cached
# unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create storage directories — these are overridden by volume mounts
# in docker-compose but must exist for bare `docker run` usage
RUN mkdir -p logs chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
