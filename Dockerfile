# Dockerfile for Iris Classifier API
# Optimized for Cloud Run deployment following Module 2 best practices

# Use slim Python image to reduce cold start time
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
# This layer is cached if requirements.txt hasn't changed
COPY requirements.txt .

# Install dependencies without caching pip files (reduces image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifact
COPY main.py .
COPY model.pkl .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# CRITICAL: Read PORT from environment variable
# Cloud Run sets PORT=8080, local development can use different ports
# Do NOT use --reload in production!
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
