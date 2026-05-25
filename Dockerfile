FROM python:3.12-slim

WORKDIR /app

# نصب dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کد پروژه
COPY . .

# جلوگیری از لود TensorFlow توسط transformers
ENV USE_TF=0
ENV USE_TORCH=1

# پورت FastAPI
EXPOSE 8000

# اجرای اپ
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
