FROM python:3.10-slim

WORKDIR /app

# Copy và cài Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code & models & static data
COPY models/ ./models/
COPY feature_extractor.py .
COPY main.py .
COPY index.html .  
COPY feature_vectors_static.csv .
COPY syscall_unique.csv .

# Bạn có thể bỏ 2 file dynamic nếu không dùng nữa,
# Nếu model chỉ dùng đặc trưng tĩnh thì không cần:
# COPY feature_vectors_syscalls_frequency_5_Cat.csv .
COPY feature_vectors_syscallsbinders_frequency_5_Cat.csv .

# Expose port và chạy uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
