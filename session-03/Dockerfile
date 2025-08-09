# Simple Scikit-Learn ML Demo - Dockerfile for Podman
# Optimized for beginner-friendly machine learning workflows

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    SCIKIT_LEARN_DATA=/tmp/scikit_learn_data

# Install system dependencies (minimal for scikit-learn)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Set working directory
WORKDIR /app

# Install Python dependencies for scikit-learn ML
RUN pip install --no-cache-dir \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    joblib==1.3.1 \
    jupyter==1.0.0

# Copy the ML demo script
COPY train.py .
# COPY test_local.py* ./

# Create directories for outputs
RUN mkdir -p outputs/models outputs/logs outputs/plots

# Create matplotlib config directory and scikit-learn data directory
RUN mkdir -p /tmp/matplotlib /tmp/scikit_learn_data && \
    chown -R mluser:mluser /tmp/matplotlib /tmp/scikit_learn_data

# Create home directory for mluser
RUN mkdir -p /home/mluser && chown -R mluser:mluser /home/mluser

# Change ownership to app user
RUN chown -R mluser:mluser /app

# Switch to app user
USER mluser

# Default command - run the ML demo
CMD ["python", "train.py"]