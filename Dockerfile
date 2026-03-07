FROM python:3.9-slim-bullseye

WORKDIR /app

# Install all system dependencies required for audio processing and building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    sox \
    libsox-fmt-mp3 \
    libsndfile1 \
    ffmpeg \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel to ensure a complete build environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies (including openai-whisper)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port your app listens on
EXPOSE 5000

# Health check (ensure your app has a /health endpoint)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
