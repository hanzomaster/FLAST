# FLAST: Fast Static Prediction of Test Flakiness
# Multi-stage Docker build for reproducible experiments

# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY py/ ./py/
COPY dataset.tgz ./
COPY README.md LICENSE ./

# Extract dataset
RUN tar zxvf dataset.tgz && rm dataset.tgz

# Create results directory
RUN mkdir -p results

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command shows help
CMD ["python", "-c", "print('FLAST: Fast Static Prediction of Test Flakiness\\n\\nAvailable experiments:\\n  python py/eff_eff.py      - RQ1 & RQ2: Effectiveness and Efficiency\\n  python py/compare_pinto.py - RQ3: Comparison with Pinto-KNN\\n  python py/params.py        - Parameter tuning experiments\\n\\nResults will be saved to the results/ directory.')"]

# Development stage with additional tools
FROM production as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov ruff mypy

# Copy test files
COPY tests/ ./tests/
COPY pyproject.toml ./

# Default command runs tests
CMD ["pytest", "tests/", "-v"]
