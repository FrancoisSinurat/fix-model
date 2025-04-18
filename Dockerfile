# Gunakan base image Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy semua file ke container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && pip install google-cloud-storage
# Expose port untuk API
CMD ["python3", "main_copy.py"]




