FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .



RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

COPY . .

ENV USE_TF=0
ENV USE_TORCH=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
