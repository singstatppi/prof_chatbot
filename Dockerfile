# Use official Python 3.11 slim image as base
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy requirements first to cache dependencies
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the rest of your source code
COPY . .

# Expose port 8000 (change if your app uses a different port)
EXPOSE 8000

# Run your chatbot.py script when container starts
CMD ["python", "chatbot.py"]
