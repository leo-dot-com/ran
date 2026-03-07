# ---- Builder stage ----
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system build dependencies (git for cloning, compilers for safety)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip, setuptools, and wheel inside the venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install torch and numpy first (pre‑built wheels)
RUN pip install --no-cache-dir torch==2.0.1 numpy==1.24.3

# Install whisper's additional dependencies (tqdm, more-itertools)
RUN pip install --no-cache-dir tqdm==4.66.1 more-itertools==10.1.0

# Clone whisper at the correct tag and install manually WITHOUT build isolation
RUN git clone --branch v20231117 https://github.com/openai/whisper.git && \
    cd whisper && \
    python setup.py install --no-deps

# Copy requirements.txt and install remaining packages (excluding openai-whisper)
COPY requirements.txt .
RUN sed -i '/openai-whisper/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app.py .

# ---- Final stage ----
FROM python:3.9-slim

WORKDIR /app

# Install runtime system dependencies (ffmpeg for audio, curl for health checks)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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
