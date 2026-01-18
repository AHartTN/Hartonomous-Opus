FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code (changes frequently, doesn't invalidate dependency cache)
COPY openai_gateway.py .

# Expose port
EXPOSE 8700

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8700/health', timeout=5.0)"

# Run the application
CMD ["python", "openai_gateway.py"]
