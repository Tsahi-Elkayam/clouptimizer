# Multi-stage build for production-ready Python application
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements/ requirements/
COPY pyproject.toml setup.py ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements/prod.txt
RUN pip install --no-cache-dir -e .

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r clouptimizer && useradd -r -g clouptimizer clouptimizer

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=clouptimizer:clouptimizer src/ src/
COPY --chown=clouptimizer:clouptimizer pyproject.toml setup.py ./

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data /app/reports && \
    chown -R clouptimizer:clouptimizer /app/logs /app/data /app/reports

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLOUPTIMIZER_ENV=production \
    CLOUPTIMIZER_LOG_DIR=/app/logs \
    CLOUPTIMIZER_DATA_DIR=/app/data \
    CLOUPTIMIZER_REPORT_DIR=/app/reports

# Switch to non-root user
USER clouptimizer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090 8080

# Default command for API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]