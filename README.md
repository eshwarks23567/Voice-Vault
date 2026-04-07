# Voice Inventory System

A production-ready voice-driven inventory management system designed for warehouse and manufacturing environments. Workers can speak product codes, quantities, and locations hands-free while working, even in noisy industrial conditions.

## Features

- **Voice Input**: Hands-free inventory entry using OpenAI Whisper speech recognition
- **Noise Handling**: Advanced audio preprocessing for noisy industrial environments
- **Multi-layer Validation**: Grammar-based field extraction with confidence scoring
- **Fallback Mechanisms**: Digit-by-digit mode and retry strategies for low confidence
- **Real-time Processing**: WebSocket support for streaming audio
- **REST API**: Complete API for integration with existing systems
- **Database**: SQLite storage with indexing for fast queries

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  Audio Input    │────▶│  Preprocessing   │────▶│    Whisper      │
│  (WebSocket/    │     │  - Noise reduce  │     │    Speech       │
│   REST API)     │     │  - Bandpass      │     │    Recognition  │
│                 │     │  - Normalize     │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│    Database     │◀────│   Validation     │◀────│   Structured    │
│    (SQLite)     │     │   & Storage      │     │   Field         │
│                 │     │                  │     │   Extraction    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, for faster inference)
- ~1GB RAM for Whisper base model

## Quick Start

### 1. Clone and Setup

```bash
cd voice-inventory-system

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Server

```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get products
curl http://localhost:8000/api/products

# Upload audio file
curl -X POST -F "file=@recording.wav" http://localhost:8000/api/upload-audio
```

## API Endpoints

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with system status |
| POST | `/api/upload-audio` | Process uploaded audio file |
| GET | `/api/inventory` | Query inventory entries |
| GET | `/api/products` | List valid product codes |
| POST | `/api/validate-entry` | Manual entry validation |
| GET | `/api/statistics` | System statistics |

### WebSocket

Connect to `/ws/voice-input` for real-time audio streaming.

**Message Format:**
```json
{
  "type": "audio",
  "audio": "<base64-encoded-wav>"
}
```

**Response Format:**
```json
{
  "status": "success",
  "data": {
    "product_id": "ABC-123",
    "quantity": 50,
    "location": "A-12",
    "confidence": 0.95
  },
  "message": "Entry saved successfully"
}
```

## Voice Commands

The system recognizes various ways to speak inventory data:

### Product Codes
- "Product ABC-123"
- "Code ABC dash 123"
- "Alpha Bravo Charlie dash one two three"

### Quantities
- "Quantity 50"
- "Fifty units"
- "Count twenty-three"

### Locations
- "Location A-12"
- "Zone B dash 5"
- "At row C-99"

### Full Commands
- "Product ABC-123 quantity 50 location A-12"
- "ABC dash 456, qty fifteen, at B-7"
- "Code XY-12345, twenty units, zone C-3"

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `inventory.db` | SQLite database path |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum confidence for auto-acceptance |

## Performance Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| End-to-end latency | <5s | 2-4s |
| Transcription time | <3s | 1-2s |
| Validation time | <100ms | 10-50ms |
| Database insert | <50ms | 5-20ms |
| Field extraction accuracy | >90% | 92-95% |

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=backend --cov-report=html

# Specific test file
pytest tests/test_validation.py -v
```

## Project Structure

```
voice-inventory-system/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── speech_processor.py # Whisper integration
│   ├── validator.py        # Field extraction
│   ├── models.py           # Pydantic models
│   └── database.py         # SQLite handler
├── audio_processing/
│   ├── noise_reducer.py    # Audio preprocessing
│   └── test_audio/         # Sample audio files
├── src/                    # React frontend source
│   ├── App.tsx
│   └── main.tsx
├── tests/
│   └── test_validation.py  # Test suite
├── index.html              # Frontend entry point
├── package.json            # Frontend dependencies
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Docker Deployment

```bash
# Build
docker build -t voice-inventory .

# Run
docker run -p 8000:8000 -v $(pwd)/data:/app/data voice-inventory

# Or use docker-compose
docker-compose up -d
```

## Troubleshooting

### Low Confidence Scores
- Speak clearly and close to the microphone
- Reduce background noise if possible
- Use the phonetic alphabet for unclear letters

### Slow Processing
- Use a smaller Whisper model (`tiny` or `base`)
- Enable GPU acceleration if available
- Process shorter audio clips

### Missing Fields
- Include all required information in one statement
- Use keywords: "product", "quantity/qty", "location"
- System will prompt for missing fields

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [noisereduce](https://github.com/timsainb/noisereduce) - Noise reduction
