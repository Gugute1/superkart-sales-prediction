FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system build dependencies required by CatBoost and scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install pinned Python dependencies first (optimizes Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app and serialized .joblib bundle into the container
COPY app.py .
COPY superkart_model_and_preprocessing_bundle.joblib .

# Expose standard FastAPI port (change to 7860 if deploying to Hugging Face Spaces)
EXPOSE 8000

# Start Uvicorn server pointing to your `superkart_api` FastAPI instance
CMD ["uvicorn", "app:superkart_api", "--host", "0.0.0.0", "--port", "8000"]
