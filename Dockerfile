# Start with a base image that has Python 3.12.3
FROM python:3.12.3-slim

# Set environment variables to prevent Python from writing pyc files and to buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (optional: adjust if you need additional system tools or packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that your application runs on (e.g., Streamlit runs on port 8501 by default)
EXPOSE 8501

# Command to run your application (replace "your_chatbot_script.py" with your actual script)
CMD ["streamlit", "run", "ollamaapp.py"]
