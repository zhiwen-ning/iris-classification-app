FROM python:3.9-slim
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型相关目录
COPY app/ ./app/                  # 复制API代码
COPY ml/registry/ ./ml/registry/  # 复制标签映射（label_map.yml）
COPY mlruns/ ./mlruns/            # 复制整个mlruns目录（包含所有模型）
COPY .env .                       # 复制环境变量文件（若需要）

# 暴露端口并启动服务
EXPOSE 5000
CMD ["python", "app/main.py"]