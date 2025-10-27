FROM python:3.9-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY app/ ./app/
COPY ml/registry/ ./ml/registry/
COPY mlruns/ ./mlruns/
COPY .env .

# Expose port and start service
EXPOSE 5000
CMD ["python", "app/main.py"]