FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (ffmpeg for audio, curl for healthcheck, compilers for safety)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip, setuptools, and wheel to the latest versions
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# (Optional) Verify setuptools is installed – for debugging
RUN pip show setuptools

# Copy requirements file
COPY requirements.txt .

# Install torch and numpy first (they are pre-built wheels, no compilation issues)
RUN pip install --no-cache-dir torch==2.0.1 numpy==1.24.3

# Install whisper without build isolation, using the already-upgraded setuptools
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

# Install remaining dependencies (flask, flask-cors, gunicorn)
RUN pip install --no-cache-dir flask==2.3.3 flask-cors==4.0.0 gunicorn==21.2.0

# Copy application code
COPY app.py .

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
