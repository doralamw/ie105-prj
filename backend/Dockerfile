# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y docker.io && \
    pip install --no-cache-dir -r requirements.txt

# Copy code & models
COPY models/ ./models/
COPY feature_extractor.py .
COPY main.py .
COPY index.html .  
COPY feature_vectors_static.csv .
COPY feature_vectors_syscalls_frequency_5_Cat.csv .
COPY feature_vectors_syscallsbinders_frequency_5_Cat.csv .
COPY syscall_unique.csv .

# Expose port và chạy uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
