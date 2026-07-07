FROM python:3.10-slim

WORKDIR /app

# System deps (FluidSynth)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fluidsynth \
    libfluidsynth3 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed data/labels \
    checkpoints outputs logs soundfonts

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
