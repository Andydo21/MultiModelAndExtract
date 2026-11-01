FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
COPY ai_extract/requirements-vision.txt /app/ai_extract/requirements-vision.txt
RUN pip install --no-cache-dir -r ai_extract/requirements-vision.txt

# Copy project files
COPY . /app/

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
