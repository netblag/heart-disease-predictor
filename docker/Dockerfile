FROM python:3.11-slim

LABEL maintainer="your-email@example.com"
LABEL description="Heart Disease Risk Predictor - ML Pipeline + API"

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/saved reports/figures

# Train model on startup if not already trained
# (In production, you'd mount pre-trained models)
RUN python main.py --no-tune

# Expose API port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]