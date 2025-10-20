# Use a minimal Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set PYTHONPATH so Flask can find your 'app' package
ENV PYTHONPATH=/app

# Expose port (Cloud Run expects 8080)
EXPOSE 8080

# Start Flask app
CMD ["python", "app/main.py"]
