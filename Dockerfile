# Use a lightweight official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for compilation and headless CV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application codebase (honoring .dockerignore)
COPY . .

# Expose the default port (Render will override this dynamically)
EXPOSE 10000

# Run Streamlit, binding to 0.0.0.0 and using Render's PORT environment variable
# Added --server.fileWatcherType="none" to prevent "inotify limit" crashes on cloud deployments
CMD streamlit run src/dashboard/app.py \
    --server.port="${PORT:-10000}" \
    --server.address="0.0.0.0" \
    --server.fileWatcherType="none" \
    --browser.gatherUsageStats=false
