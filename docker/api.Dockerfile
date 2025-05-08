FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies - with selective versions to avoid mismatches
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install optional dependencies for production
RUN pip install gunicorn psutil

# Copy source code (or use volume mount in docker-compose)
COPY . .

# Use the slim version of the API to ensure reliable startup
CMD ["python", "slim_api.py"] 
