# Use a slim Python image
FROM python:3.11-slim

# Install system dependencies that OpenCV needs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Run streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
