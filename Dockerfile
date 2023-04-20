FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining files
COPY . .

# Expose port
EXPOSE 5000

# Run command
CMD ["python", "api.py"]
