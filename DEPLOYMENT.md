# Deployment Guide

## Voice Inventory System - Production Deployment

This guide covers deploying the Voice Inventory System to production environments.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Environment Variables](#environment-variables)
5. [Monitoring](#monitoring)
6. [Scaling](#scaling)
7. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.9+
- FFmpeg (for audio processing)
- 4GB+ RAM
- CUDA toolkit (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd voice-inventory-system

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from backend.database import get_database; db = get_database(); db.seed_sample_products(50)"

# Start development server
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Development Tools

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend --cov-report=html

# Format code
black backend/ tests/

# Lint
flake8 backend/ tests/
```

---

## Docker Deployment

### Build Image

```bash
# Build the Docker image
docker build -t voice-inventory:latest .

# Build with specific Whisper model
docker build --build-arg WHISPER_MODEL=small -t voice-inventory:small .
```

### Run Container

```bash
# Basic run
docker run -p 8000:8000 voice-inventory:latest

# With persistent data
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/audio_files:/app/audio_files \
  voice-inventory:latest

# With environment variables
docker run -p 8000:8000 \
  -e WHISPER_MODEL=small \
  -e CONFIDENCE_THRESHOLD=0.80 \
  voice-inventory:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Docker Compose Configuration

Edit `docker-compose.yml` for production:

```yaml
services:
  backend:
    environment:
      - WHISPER_MODEL=medium  # or 'small' for faster inference
      - CONFIDENCE_THRESHOLD=0.85
      - DATABASE_PATH=/app/data/inventory.db
    deploy:
      resources:
        limits:
          memory: 8G  # Adjust based on model size
          cpus: '4'
```

---

## Cloud Deployment

### Render.com

1. **Create New Web Service**
   - Connect GitHub repository
   - Select Docker runtime
   - Set environment variables

2. **Environment Variables**
   ```
   WHISPER_MODEL=small
   CONFIDENCE_THRESHOLD=0.85
   DATABASE_PATH=/app/data/inventory.db
   ```

3. **Disk Configuration**
   - Add persistent disk for `/app/data`

4. **Start Command**
   ```
   uvicorn backend.app:app --host 0.0.0.0 --port $PORT
   ```

### Railway.app

1. **Deploy from GitHub**
   ```bash
   railway login
   railway init
   railway up
   ```

2. **Add Variables**
   ```bash
   railway variables set WHISPER_MODEL=small
   railway variables set CONFIDENCE_THRESHOLD=0.85
   ```

3. **Configure Volume**
   - Add volume for persistent storage

### AWS (ECS/Fargate)

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name voice-inventory
   ```

2. **Push Image**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   docker tag voice-inventory:latest <account>.dkr.ecr.us-east-1.amazonaws.com/voice-inventory:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/voice-inventory:latest
   ```

3. **Create Task Definition**
   ```json
   {
     "family": "voice-inventory",
     "containerDefinitions": [
       {
         "name": "backend",
         "image": "<account>.dkr.ecr.us-east-1.amazonaws.com/voice-inventory:latest",
         "memory": 8192,
         "cpu": 2048,
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {"name": "WHISPER_MODEL", "value": "small"},
           {"name": "CONFIDENCE_THRESHOLD", "value": "0.85"}
         ]
       }
     ]
   }
   ```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `inventory.db` | Path to SQLite database |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum confidence for auto-accept |
| `MAX_CONNECTIONS` | `50` | Max WebSocket connections |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `RATE_LIMIT` | `10/second` | Rate limit per client |

### Model Size Comparison

| Model | Size | RAM | Speed | Accuracy |
|-------|------|-----|-------|----------|
| tiny | 39M | 1GB | ~1s | ~85% |
| base | 74M | 1GB | ~1.5s | ~88% |
| small | 244M | 2GB | ~2s | ~91% |
| medium | 769M | 4GB | ~4s | ~94% |
| large | 1550M | 8GB | ~8s | ~96% |

---

## Monitoring

### Prometheus Metrics

Add Prometheus metrics endpoint:

```python
# In app.py
from prometheus_client import Counter, Histogram, generate_latest

REQUESTS = Counter('voice_inventory_requests', 'Total requests')
LATENCY = Histogram('voice_inventory_latency', 'Request latency')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Grafana Dashboard

Import dashboard with these panels:
- Request rate
- Latency percentiles (p50, p95, p99)
- Error rate
- Confidence score distribution
- Active WebSocket connections

### Logging

Configure structured logging:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger().addHandler(handler)
```

### Health Checks

The `/health` endpoint provides:
- Model loading status
- Database connectivity
- Uptime
- Statistics

Use for load balancer health checks:

```nginx
location /health {
    proxy_pass http://backend:8000/health;
    proxy_connect_timeout 5s;
    proxy_read_timeout 5s;
}
```

---

## Scaling

### Horizontal Scaling

1. **Stateless Design**
   - Use external database (PostgreSQL)
   - Use Redis for session caching
   - Store audio files in S3/MinIO

2. **Load Balancer Configuration**
   ```nginx
   upstream backend {
       least_conn;
       server backend1:8000;
       server backend2:8000;
       server backend3:8000;
   }
   ```

3. **WebSocket Sticky Sessions**
   ```nginx
   upstream websocket {
       ip_hash;
       server backend1:8000;
       server backend2:8000;
   }
   ```

### Vertical Scaling

- **CPU**: More cores for parallel audio processing
- **RAM**: Larger Whisper models need more RAM
- **GPU**: CUDA acceleration for faster inference

### GPU Deployment

```dockerfile
# Use CUDA base image
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install PyTorch with CUDA
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
```

```bash
# Run with GPU
docker run --gpus all -p 8000:8000 voice-inventory:cuda
```

---

## Troubleshooting

### Common Issues

**Model Loading Slow**
- First request downloads model weights (~1.5GB for medium)
- Use smaller model for faster startup
- Pre-download model in Docker build

**Out of Memory**
- Reduce model size (small instead of medium)
- Limit concurrent connections
- Add swap space

**WebSocket Disconnections**
- Check nginx timeout settings
- Increase `proxy_read_timeout`
- Implement heartbeat ping/pong

**Low Confidence Scores**
- Check audio quality
- Reduce background noise
- Adjust `CONFIDENCE_THRESHOLD`

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG uvicorn backend.app:app --reload

# Test transcription
python -c "
from backend.speech_processor import get_recognizer
r = get_recognizer('tiny')
result = r.transcribe_audio('test.wav')
print(result)
"
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run processing
result = process_audio_file('test.wav')

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## Security Considerations

1. **API Authentication**
   - Implement JWT or API keys
   - Rate limit by user/API key

2. **Input Validation**
   - Validate audio file sizes
   - Sanitize transcription output
   - Validate product codes against database

3. **Network Security**
   - Use HTTPS in production
   - Configure firewall rules
   - Use VPC for internal services

4. **Data Privacy**
   - Don't store audio files longer than needed
   - Anonymize logs
   - Implement data retention policies
