FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (ffmpeg, compilers, git)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# (Optional) Verify setuptools is importable
RUN python -c "import setuptools; print('setuptools version:', setuptools.__version__)"

# Install torch and numpy first (they are pre‑built wheels)
RUN pip install --no-cache-dir torch==2.0.1 numpy==1.24.3

# Clone whisper and install manually using setuptools
RUN git clone https://github.com/openai/whisper.git && \
    cd whisper && \
    git checkout 20231117 && \
    python setup.py install

# Copy requirements.txt and install remaining packages
COPY requirements.txt .
# Exclude openai-whisper from requirements.txt (already installed)
RUN grep -v "openai-whisper" requirements.txt | xargs pip install --no-cache-dir

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
