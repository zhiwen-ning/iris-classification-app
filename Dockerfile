FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY ml/registry/ ./ml/registry/
COPY mlruns/0/e7f8a9b/artifacts/improved_model ./mlruns/0/e7f8a9b/artifacts/improved_model
COPY .env .
EXPOSE 5000
CMD ["python", "app/main.py"]