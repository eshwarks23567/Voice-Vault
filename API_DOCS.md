# API Documentation

## Voice Inventory System API Reference

Base URL: `http://localhost:8000`

## Authentication

Currently, the API does not require authentication. For production deployments, implement OAuth2 or API key authentication.

---

## REST Endpoints

### Health Check

Check system health and status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "db_connected": true,
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "statistics": {
    "total_entries": 150,
    "total_products": 50,
    "average_confidence": 0.92,
    "success_rate": 0.95
  }
}
```

**Status Codes:**
- `200 OK` - System is healthy
- `503 Service Unavailable` - System is degraded

---

### Upload Audio

Upload and process an audio file for transcription and validation.

**Endpoint:** `POST /api/upload-audio`

**Content-Type:** `multipart/form-data`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | Audio file (WAV, MP3, FLAC) |
| language | string | No | Language code (default: "en") |

**cURL Example:**
```bash
curl -X POST \
  -F "file=@recording.wav" \
  -F "language=en" \
  http://localhost:8000/api/upload-audio
```

**Success Response (200):**
```json
{
  "status": "success",
  "message": "Processing complete",
  "transcription": {
    "text": "Product ABC-123 quantity 50 location A-12",
    "confidence": 0.95,
    "language": "en",
    "duration": 2.5
  },
  "validation": {
    "status": "success",
    "data": {
      "product_id": "ABC-123",
      "quantity": 50,
      "location": "A-12",
      "confidence": 0.95,
      "timestamp": "2026-02-04T10:30:00Z"
    }
  },
  "entry_id": 123,
  "processing_time_ms": 2500.5
}
```

**Low Confidence Response (200):**
```json
{
  "status": "low_confidence",
  "message": "Confidence too low (70%). Please speak more clearly and repeat.",
  "transcription": {
    "text": "Product ABC something quantity...",
    "confidence": 0.70
  },
  "validation": {
    "status": "low_confidence",
    "retry": true,
    "partial_extraction": {
      "product_id": "ABC-123",
      "quantity": null,
      "location": null
    }
  }
}
```

**Error Response (400):**
```json
{
  "status": "error",
  "error_code": "UNSUPPORTED_FORMAT",
  "message": "Unsupported audio format: audio/aac"
}
```

---

### Get Inventory

Query inventory entries with optional filters.

**Endpoint:** `GET /api/inventory`

**Query Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| product_code | string | No | Filter by product code |
| location | string | No | Filter by location |
| start_date | datetime | No | Filter entries after this date |
| end_date | datetime | No | Filter entries before this date |
| limit | int | No | Max results (1-1000, default: 100) |
| offset | int | No | Skip results (default: 0) |

**cURL Example:**
```bash
curl "http://localhost:8000/api/inventory?location=A-12&limit=10"
```

**Response:**
```json
{
  "entries": [
    {
      "id": 123,
      "product_code": "ABC-123",
      "product_name": "Widget ABC-123",
      "category": "Electronics",
      "quantity": 50,
      "location": "A-12",
      "confidence_score": 0.95,
      "raw_transcription": "Product ABC-123 quantity 50 location A-12",
      "verified": false,
      "created_at": "2026-02-04T10:30:00Z"
    }
  ],
  "total_count": 1,
  "limit": 10,
  "offset": 0,
  "has_more": false
}
```

---

### Get Products

List all valid product codes.

**Endpoint:** `GET /api/products`

**Query Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| active_only | bool | No | Only active products (default: true) |

**Response:**
```json
{
  "products": [
    {
      "id": 1,
      "product_code": "ABC-123",
      "name": "Widget ABC-123",
      "category": "Electronics",
      "active": true
    },
    {
      "id": 2,
      "product_code": "XY-45678",
      "name": "Gadget XY-45678",
      "category": "Hardware",
      "active": true
    }
  ],
  "total_count": 50
}
```

---

### Validate Entry

Manually validate and submit an inventory entry.

**Endpoint:** `POST /api/validate-entry`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "product_id": "ABC-123",
  "quantity": 50,
  "location": "A-12"
}
```

**Success Response:**
```json
{
  "status": "success",
  "message": "Entry added successfully",
  "entry_id": 124,
  "product_details": {
    "id": 1,
    "product_code": "ABC-123",
    "name": "Widget ABC-123",
    "category": "Electronics"
  }
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Product XYZ-999 not found in database"
}
```

---

### Get Statistics

Get system statistics and metrics.

**Endpoint:** `GET /api/statistics`

**Response:**
```json
{
  "total_entries": 1500,
  "total_products": 50,
  "average_confidence": 0.923,
  "entries_today": 45,
  "total_failures": 75,
  "success_rate": 0.952
}
```

---

## WebSocket API

### Voice Input Stream

Real-time audio streaming for voice input.

**Endpoint:** `ws://localhost:8000/ws/voice-input`

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/voice-input');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('Response:', response);
};
```

### Message Types

#### Send Audio

Send audio data for processing:

```json
{
  "type": "audio",
  "audio": "<base64-encoded-wav-data>"
}
```

#### Send Command

Send control commands:

```json
{
  "type": "command",
  "command": "reset"  // or "status"
}
```

#### Digit-by-Digit Mode

Enter digit-by-digit input mode:

```json
{
  "type": "digit_by_digit",
  "field": "product_id"
}
```

Then send individual characters:

```json
{
  "type": "digit_by_digit",
  "character": "A"
}
```

### Response Types

#### Connection Acknowledged

```json
{
  "type": "connected",
  "message": "Connected to Voice Inventory System",
  "client_id": "12345"
}
```

#### Processing Result

```json
{
  "type": "result",
  "status": "success",
  "text": "Product ABC-123 quantity 50 location A-12",
  "confidence": 0.95,
  "data": {
    "product_id": "ABC-123",
    "quantity": 50,
    "location": "A-12"
  },
  "entry_id": 125,
  "processing_time_ms": 2100
}
```

#### Retry Request

```json
{
  "type": "result",
  "status": "low_confidence",
  "confidence": 0.65,
  "retry_strategy": {
    "action": "slower",
    "message": "Please speak more slowly and clearly.",
    "attempt": 2,
    "max_attempts": 4
  }
}
```

#### Error

```json
{
  "type": "error",
  "status": "error",
  "message": "Audio processing failed"
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `AUDIO_TOO_SHORT` | Audio duration less than 0.5 seconds |
| `AUDIO_TOO_LONG` | Audio duration exceeds 5 minutes |
| `UNSUPPORTED_FORMAT` | Audio format not supported |
| `TRANSCRIPTION_FAILED` | Whisper transcription error |
| `LOW_CONFIDENCE` | Confidence below threshold |
| `MISSING_FIELDS` | Required fields not extracted |
| `INVALID_PRODUCT` | Product code not in database |
| `INVALID_LOCATION` | Location format invalid |
| `INVALID_QUANTITY` | Quantity out of range (1-9999) |
| `DATABASE_ERROR` | Database operation failed |
| `RATE_LIMITED` | Too many requests |
| `CONNECTION_TIMEOUT` | WebSocket timeout |

---

## Rate Limits

- REST endpoints: 30 requests/second per client
- Audio upload: 10 requests/second per client
- WebSocket: 50 concurrent connections max

---

## Python Client Example

```python
import requests
import base64

# Upload audio file
def upload_audio(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/api/upload-audio',
            files={'file': f}
        )
    return response.json()

# Query inventory
def get_inventory(location=None):
    params = {}
    if location:
        params['location'] = location
    response = requests.get(
        'http://localhost:8000/api/inventory',
        params=params
    )
    return response.json()

# Manual entry
def add_entry(product_id, quantity, location):
    response = requests.post(
        'http://localhost:8000/api/validate-entry',
        json={
            'product_id': product_id,
            'quantity': quantity,
            'location': location
        }
    )
    return response.json()

# Example usage
result = upload_audio('recording.wav')
print(f"Status: {result['status']}")
if result['status'] == 'success':
    print(f"Product: {result['validation']['data']['product_id']}")
```

---

## JavaScript/WebSocket Example

```javascript
class VoiceInventoryClient {
  constructor(url = 'ws://localhost:8000/ws/voice-input') {
    this.url = url;
    this.ws = null;
    this.onResult = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => resolve();
      this.ws.onerror = (err) => reject(err);
      
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (this.onResult) {
          this.onResult(data);
        }
      };
    });
  }

  sendAudio(audioBlob) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(',')[1];
        this.ws.send(JSON.stringify({
          type: 'audio',
          audio: base64
        }));
        resolve();
      };
      reader.readAsDataURL(audioBlob);
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage
const client = new VoiceInventoryClient();
client.onResult = (data) => console.log('Result:', data);
await client.connect();
await client.sendAudio(audioBlob);
```
