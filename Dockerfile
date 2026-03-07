# ---- Builder stage (full Python image) ----
FROM python:3.9 AS builder

WORKDIR /app

# Install system dependencies required for audio processing and building
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip, setuptools, and wheel (pin setuptools to a version known to work)
RUN pip install --no-cache-dir --upgrade pip setuptools==69.5.1 wheel

# Install torch and numpy first (pre‑built wheels)
RUN pip install --no-cache-dir torch==2.0.1 numpy==1.24.3

# Install openai-whisper directly from GitHub using the correct tag
# --no-build-isolation ensures it uses the venv's setuptools
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/openai/whisper.git@v20231117

# Copy requirements.txt and install remaining packages (flask, etc.)
COPY requirements.txt .
# Remove openai-whisper from requirements.txt (already installed)
RUN sed -i '/openai-whisper/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app.py .

# ---- Final stage (slim image) ----
FROM python:3.9-slim-bullseye

WORKDIR /app

# Install runtime system dependencies (ffmpeg for audio, curl for health checks)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
# Copy your application code
COPY --from=builder /app/app.py .

# Set the PATH to use the venv's Python
ENV PATH="/opt/venv/bin:$PATH"

# Create a non‑root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Expose the port your app listens on
EXPOSE 5000

# Health check (ensure your app has a /health endpoint)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
