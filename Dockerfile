
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi \
    uvicorn[standard] \
    python-multipart

# Copy source code
COPY src/ ./src/
COPY app.py .
COPY templates/ ./templates/
COPY static/ /static/

# Copy the PRE-TRAINED model
COPY artifacts/model_training/ ./artifacts/model_training/

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
