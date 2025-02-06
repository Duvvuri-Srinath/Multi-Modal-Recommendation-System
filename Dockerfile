# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (make sure this matches the API port)
EXPOSE 5000

# Command to run the API using gunicorn for production-grade serving
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.recommendation_api:app"]