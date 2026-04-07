"""
FastAPI Backend for Voice Inventory System

Main application with WebSocket support, REST endpoints,
middleware for CORS/logging/rate-limiting, and fallback handling.
"""

import os
import sys
import asyncio
import logging
import tempfile
import time
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import json

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException,
    UploadFile, File, Query, Depends, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
import aiofiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cachetools import TTLCache

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import (
    ProcessingStatus, TranscriptionResult, ValidationResult,
    InventoryEntryData, AudioUploadResponse, ManualEntryRequest,
    ManualEntryResponse, InventoryListResponse, ProductListResponse,
    HealthCheckResponse, WebSocketResponse, StatisticsResponse,
    ErrorResponse, ErrorCodes, ProductInfo, InventoryEntry
)
from backend.database import InventoryDatabase, get_database
from backend.validator import StructuredFieldExtractor
from backend.speech_processor import SpeechRecognizer, get_recognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Product validation cache (TTL: 1 hour)
product_cache = TTLCache(maxsize=1000, ttl=3600)

# Global instances
speech_recognizer: Optional[SpeechRecognizer] = None
database: Optional[InventoryDatabase] = None
extractor: Optional[StructuredFieldExtractor] = None
startup_time: float = 0

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections."""
    
    MAX_CONNECTIONS = 50
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Accept a new WebSocket connection."""
        async with self._lock:
            if len(self.active_connections) >= self.MAX_CONNECTIONS:
                logger.warning(f"Max connections reached, rejecting {client_id}")
                return False
            
            await websocket.accept()
            self.active_connections[client_id] = websocket
            logger.info(f"Client connected: {client_id}")
            return True
    
    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                logger.info(f"Client disconnected: {client_id}")
    
    async def send_message(self, client_id: str, message: dict):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    def get_connection_count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()


class FallbackHandler:
    """
    Handles fallback mechanisms for low-confidence transcriptions.
    
    Provides:
    - Digit-by-digit mode for spelling out codes
    - Retry strategies with exponential backoff
    - Failure logging for debugging
    """
    
    MAX_RETRIES = 4
    
    def __init__(self, database: InventoryDatabase, extractor: StructuredFieldExtractor):
        self.database = database
        self.extractor = extractor
        self.retry_delays = [0, 1, 2, 5]  # Exponential backoff
    
    def handle_low_confidence(
        self,
        field_name: str,
        transcription: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Handle low confidence for a specific field.
        
        Args:
            field_name: Which field failed ('product_id', 'quantity', 'location')
            transcription: The original transcription
            confidence: The confidence score
        
        Returns:
            Dict with retry instructions
        """
        prompts = {
            'product_id': "Please spell out the product code letter by letter. For example: A-B-C dash 1-2-3",
            'quantity': "Please say the quantity as individual digits. For example: five-zero for 50",
            'location': "Please spell out the location. For example: Alpha dash 1-2"
        }
        
        return {
            'status': ProcessingStatus.RETRY,
            'field': field_name,
            'message': prompts.get(field_name, f"Please repeat the {field_name} more clearly"),
            'mode': 'digit_by_digit',
            'original_transcription': transcription,
            'confidence': confidence
        }
    
    async def digit_by_digit_mode(
        self,
        websocket: WebSocket,
        field_name: str,
        timeout: float = 30.0
    ) -> Optional[str]:
        """
        Enable digit-by-digit listening mode.
        
        Accepts individual characters and assembles them into a value.
        
        Args:
            websocket: WebSocket connection
            field_name: Which field is being entered
            timeout: Maximum time to wait for complete entry
        
        Returns:
            Assembled value or None if timeout/cancelled
        """
        assembled = []
        start_time = time.time()
        
        await websocket.send_json({
            'type': 'mode_change',
            'mode': 'digit_by_digit',
            'field': field_name,
            'message': f'Entering digit-by-digit mode for {field_name}. Say each character, then "done" when complete.'
        })
        
        while time.time() - start_time < timeout:
            try:
                # Wait for next message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=10.0
                )
                
                char_input = data.get('character', '').upper().strip()
                
                if char_input in ['DONE', 'FINISHED', 'COMPLETE']:
                    break
                
                if char_input in ['BACK', 'BACKSPACE', 'DELETE', 'UNDO']:
                    if assembled:
                        removed = assembled.pop()
                        await websocket.send_json({
                            'type': 'char_removed',
                            'removed': removed,
                            'current': ''.join(assembled)
                        })
                    continue
                
                if char_input == 'CLEAR':
                    assembled.clear()
                    await websocket.send_json({
                        'type': 'cleared',
                        'current': ''
                    })
                    continue
                
                # Parse the character
                parsed = self.extractor.parse_digit_by_digit(char_input)
                if parsed:
                    assembled.append(parsed)
                    await websocket.send_json({
                        'type': 'char_added',
                        'added': parsed,
                        'current': ''.join(assembled)
                    })
            
            except asyncio.TimeoutError:
                await websocket.send_json({
                    'type': 'timeout_warning',
                    'message': 'No input received. Say a character or "done" to complete.'
                })
        
        result = ''.join(assembled)
        
        await websocket.send_json({
            'type': 'mode_complete',
            'result': result,
            'field': field_name
        })
        
        return result if result else None
    
    def get_retry_strategy(
        self,
        attempt_count: int,
        error_type: str
    ) -> Dict[str, Any]:
        """
        Get the retry strategy based on attempt count and error type.
        
        Attempt progression:
        1. Normal transcription (try again)
        2. Ask to speak slower
        3. Switch to digit-by-digit mode
        4. Suggest manual keyboard entry
        
        Args:
            attempt_count: Number of attempts so far
            error_type: Type of error encountered
        
        Returns:
            Dict with retry strategy
        """
        strategies = [
            {
                'action': 'retry',
                'message': 'Please try again. Speak clearly near the microphone.',
                'delay': 0
            },
            {
                'action': 'slower',
                'message': 'Please speak more slowly and clearly. Pause between words.',
                'delay': 1
            },
            {
                'action': 'digit_by_digit',
                'message': 'Switching to digit-by-digit mode. Spell out each character.',
                'delay': 2
            },
            {
                'action': 'manual',
                'message': 'Voice input unsuccessful. Please use keyboard entry.',
                'delay': 0
            }
        ]
        
        idx = min(attempt_count - 1, len(strategies) - 1)
        strategy = strategies[idx]
        strategy['attempt'] = attempt_count
        strategy['max_attempts'] = self.MAX_RETRIES
        
        return strategy
    
    async def log_failure(
        self,
        audio_path: Optional[str],
        transcription: str,
        error_type: str,
        error_message: str,
        confidence: Optional[float] = None,
        attempt_count: int = 1
    ):
        """
        Log a failed attempt for debugging and improvement.
        
        Args:
            audio_path: Path to the audio file
            transcription: The failed transcription
            error_type: Type of error
            error_message: Detailed error message
            confidence: Confidence score if available
            attempt_count: Number of attempts made
        """
        try:
            self.database.log_failure(
                error_type=error_type,
                error_message=error_message,
                audio_path=audio_path,
                transcription=transcription,
                confidence=confidence,
                attempt_count=attempt_count
            )
        except Exception as e:
            logger.error(f"Failed to log failure: {e}")


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global speech_recognizer, database, extractor, startup_time
    
    startup_time = time.time()
    logger.info("Starting Voice Inventory System...")
    
    # Initialize database
    db_path = os.environ.get('DATABASE_PATH', 'inventory.db')
    database = InventoryDatabase(db_path)
    database.initialize_database()
    
    # Seed sample products if database is empty
    product_count = len(database.get_all_product_codes())
    if product_count == 0:
        logger.info("Seeding sample products...")
        database.seed_sample_products(50)
    
    # Initialize validator
    extractor = StructuredFieldExtractor()
    
    # Load speech recognition model
    # Use 'base' for fast inference (16x faster than medium)
    model_name = os.environ.get('WHISPER_MODEL', 'base')
    logger.info(f"Loading Whisper {model_name} model...")
    try:
        speech_recognizer = get_recognizer(model_name)
        logger.info("Speech recognition model loaded")
    except Exception as e:
        logger.error(f"Failed to load speech model: {e}")
        speech_recognizer = None
    
    logger.info("Voice Inventory System started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if database:
        database.close()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Voice Inventory System",
    description="Voice-driven inventory management for warehouse/manufacturing",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - Time: {process_time:.2f}ms"
    )
    
    return response


# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'frontend', 'index.html'
    )
    if os.path.exists(frontend_path):
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


# Helper functions
async def process_audio_file(
    file_path: str,
    language: str = 'en'
) -> Dict[str, Any]:
    """
    Process an audio file through the complete pipeline.
    
    Returns:
        Dict with transcription and validation results
    """
    global speech_recognizer, extractor, database
    
    if not speech_recognizer:
        raise HTTPException(500, "Speech recognition model not loaded")
    
    start_time = time.time()
    
    # Transcribe audio
    transcription = speech_recognizer.transcribe_audio(file_path, language)
    transcription_time = time.time() - start_time
    
    # Validate and extract fields
    validation = extractor.validate_and_structure(transcription)
    validation_time = time.time() - start_time - transcription_time
    
    # If successful, save to database
    entry_id = None
    if validation['status'] == 'success':
        data = validation['data']
        
        # Check product exists
        if database.product_exists(data['product_id']):
            entry_id = database.insert_entry(
                product_code=data['product_id'],
                quantity=data['quantity'],
                location=data['location'],
                confidence=data['confidence'],
                transcription=transcription['text'],
                audio_path=file_path
            )
        else:
            validation['status'] = 'incomplete'
            validation['message'] = f"Product {data['product_id']} not found in database"
    
    total_time = time.time() - start_time
    
    return {
        'transcription': transcription,
        'validation': validation,
        'entry_id': entry_id,
        'processing_times': {
            'transcription_ms': transcription_time * 1000,
            'validation_ms': validation_time * 1000,
            'total_ms': total_time * 1000
        }
    }


# WebSocket endpoint
@app.websocket("/ws/voice-input")
async def websocket_voice_input(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice input.
    
    Protocol:
    1. Client connects and receives acknowledgment
    2. Client sends audio chunks as base64
    3. Server processes and responds with results
    4. Low confidence triggers retry flow
    """
    client_id = str(id(websocket))
    
    if not await manager.connect(websocket, client_id):
        await websocket.close(code=1013, reason="Max connections reached")
        return
    
    fallback_handler = FallbackHandler(database, extractor)
    attempt_count = 0
    
    try:
        # Send welcome message
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to Voice Inventory System',
            'client_id': client_id
        })
        
        while True:
            # Receive message
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60.0  # Connection timeout
                )
            except asyncio.TimeoutError:
                await websocket.send_json({
                    'type': 'ping',
                    'message': 'Connection alive check'
                })
                continue
            
            msg_type = data.get('type', 'audio')
            
            if msg_type == 'pong':
                continue
            
            if msg_type == 'audio':
                # Process audio data
                audio_base64 = data.get('audio')
                
                if not audio_base64:
                    await websocket.send_json({
                        'type': 'error',
                        'status': ProcessingStatus.ERROR,
                        'message': 'No audio data received'
                    })
                    continue
                
                attempt_count += 1
                
                try:
                    # Decode audio
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    # Save to temp file with correct extension for browser audio
                    with tempfile.NamedTemporaryFile(
                        suffix='.webm',
                        delete=False
                    ) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    logger.info(f"Processing audio file: {tmp_path}")
                    
                    # Process audio
                    result = await process_audio_file(tmp_path)
                    
                    # Build response
                    validation = result['validation']
                    
                    response = {
                        'type': 'result',
                        'status': validation['status'],
                        'message': validation.get('message', ''),
                        'confidence': result['transcription']['confidence'],
                        'text': result['transcription']['text'],
                        'processing_time_ms': result['processing_times']['total_ms']
                    }
                    
                    if validation['status'] == 'success':
                        response['data'] = validation['data']
                        response['entry_id'] = result['entry_id']
                        attempt_count = 0  # Reset on success
                    
                    elif validation['status'] == 'low_confidence':
                        # Determine retry strategy
                        strategy = fallback_handler.get_retry_strategy(
                            attempt_count,
                            'low_confidence'
                        )
                        response['retry_strategy'] = strategy
                        
                        # Log failure
                        await fallback_handler.log_failure(
                            tmp_path,
                            result['transcription']['text'],
                            'low_confidence',
                            f"Confidence: {result['transcription']['confidence']:.2%}",
                            result['transcription']['confidence'],
                            attempt_count
                        )
                    
                    elif validation['status'] == 'incomplete':
                        response['missing_fields'] = validation.get('missing_fields', [])
                        response['found_fields'] = validation.get('found_fields', {})
                    
                    logger.info(f"Sending response to client: {response['status']}, text: {response.get('text', '')[:50]}")
                    await websocket.send_json(response)
                    logger.info("Response sent successfully")
                    
                    # Cleanup temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                
                except Exception as e:
                    logger.error(f"Audio processing error: {e}", exc_info=True)
                    error_msg = str(e)
                    if "Failed to load audio file" in error_msg or "ffmpeg" in error_msg.lower():
                        error_msg = "Audio processing failed. Please ensure ffmpeg is installed and try again."
                    await websocket.send_json({
                        'type': 'error',
                        'status': 'error',
                        'message': error_msg
                    })
            
            elif msg_type == 'digit_by_digit':
                # Handle digit-by-digit mode
                field_name = data.get('field', 'product_id')
                result = await fallback_handler.digit_by_digit_mode(
                    websocket,
                    field_name,
                    timeout=30.0
                )
                
                if result:
                    await websocket.send_json({
                        'type': 'digit_result',
                        'field': field_name,
                        'value': result,
                        'status': ProcessingStatus.SUCCESS
                    })
            
            elif msg_type == 'command':
                # Handle commands
                command = data.get('command')
                
                if command == 'reset':
                    attempt_count = 0
                    await websocket.send_json({
                        'type': 'reset',
                        'message': 'Session reset'
                    })
                
                elif command == 'status':
                    await websocket.send_json({
                        'type': 'status',
                        'attempt_count': attempt_count,
                        'model_loaded': speech_recognizer is not None,
                        'connections': manager.get_connection_count()
                    })
    
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(client_id)


# REST Endpoints
@app.post("/api/upload-audio", response_model=AudioUploadResponse)
@limiter.limit("10/second")
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    language: str = Query(default='en')
):
    """
    Upload and process an audio file.
    
    Accepts WAV, MP3, FLAC, and other common audio formats.
    Returns transcription, validation, and database entry ID if successful.
    """
    # Validate file type
    allowed_types = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/flac', 'audio/x-flac',
        'audio/ogg', 'audio/webm'
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            400,
            f"Unsupported audio format: {file.content_type}"
        )
    
    start_time = time.time()
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(file.filename)[1] or '.wav',
        delete=False
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = await process_audio_file(tmp_path, language)
        
        return AudioUploadResponse(
            status=ProcessingStatus(result['validation']['status']),
            message=result['validation'].get('message', 'Processing complete'),
            transcription=TranscriptionResult(
                text=result['transcription']['text'],
                confidence=result['transcription']['confidence'],
                language=result['transcription']['language'],
                duration=result['transcription']['duration'],
                segments=[],  # Simplified for response
                low_confidence_words=result['transcription'].get('low_confidence_words', [])
            ),
            validation=ValidationResult(
                status=ProcessingStatus(result['validation']['status']),
                message=result['validation'].get('message', ''),
                raw_text=result['transcription']['text'],
                confidence=result['transcription']['confidence'],
                missing_fields=result['validation'].get('missing_fields', []),
                found_fields=result['validation'].get('found_fields', {})
            ),
            entry_id=result.get('entry_id'),
            processing_time_ms=result['processing_times']['total_ms']
        )
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.get("/api/inventory", response_model=InventoryListResponse)
@limiter.limit("30/second")
async def get_inventory(
    request: Request,
    product_code: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """
    Get inventory entries with optional filters.
    
    Supports filtering by product code, location, and date range.
    Includes pagination with limit/offset.
    """
    entries, total = database.get_inventory_entries(
        product_code=product_code,
        location=location,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    
    return InventoryListResponse(
        entries=[
            InventoryEntry(
                id=e['id'],
                product_code=e['product_code'],
                product_name=e.get('product_name'),
                category=e.get('category'),
                quantity=e['quantity'],
                location=e['location'],
                confidence_score=e['confidence_score'],
                raw_transcription=e.get('raw_transcription'),
                audio_path=e.get('audio_path'),
                verified=bool(e.get('verified', False)),
                created_at=datetime.fromisoformat(e['created_at']) if isinstance(e['created_at'], str) else e['created_at']
            )
            for e in entries
        ],
        total_count=total,
        limit=limit,
        offset=offset,
        has_more=offset + limit < total
    )


@app.get("/api/products", response_model=ProductListResponse)
@limiter.limit("30/second")
async def get_products(
    request: Request,
    active_only: bool = Query(default=True)
):
    """
    Get list of valid product codes.
    
    Used for auto-complete suggestions and validation.
    """
    codes = database.get_all_product_codes(active_only=active_only)
    
    products = []
    for code in codes:
        # Use cache
        if code in product_cache:
            products.append(product_cache[code])
        else:
            details = database.get_product_details(code)
            if details:
                product = ProductInfo(**details)
                product_cache[code] = product
                products.append(product)
    
    return ProductListResponse(
        products=products,
        total_count=len(products)
    )


@app.post("/api/validate-entry", response_model=ManualEntryResponse)
@limiter.limit("10/second")
async def validate_entry(
    request: Request,
    entry: ManualEntryRequest
):
    """
    Manually validate and submit an inventory entry.
    
    Used as fallback when voice input fails or for manual corrections.
    """
    # Validate product exists
    if not database.product_exists(entry.product_id):
        return ManualEntryResponse(
            status=ProcessingStatus.ERROR,
            message=f"Product {entry.product_id} not found in database"
        )
    
    # Validate location format
    if not extractor.extract_location(entry.location):
        return ManualEntryResponse(
            status=ProcessingStatus.ERROR,
            message=f"Invalid location format: {entry.location}"
        )
    
    # Insert entry
    try:
        entry_id = database.insert_entry(
            product_code=entry.product_id,
            quantity=entry.quantity,
            location=entry.location,
            confidence=1.0,  # Manual entry = 100% confidence
            transcription=f"Manual entry: {entry.product_id} x{entry.quantity} @ {entry.location}"
        )
        
        product_details = database.get_product_details(entry.product_id)
        
        return ManualEntryResponse(
            status=ProcessingStatus.SUCCESS,
            message="Entry added successfully",
            entry_id=entry_id,
            product_details=ProductInfo(**product_details) if product_details else None
        )
    
    except Exception as e:
        logger.error(f"Failed to insert entry: {e}")
        return ManualEntryResponse(
            status=ProcessingStatus.ERROR,
            message=str(e)
        )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status including model loading and database connectivity.
    """
    uptime = time.time() - startup_time
    
    # Check database connection
    db_connected = False
    try:
        stats = database.get_statistics()
        db_connected = True
    except:
        stats = None
    
    return HealthCheckResponse(
        status="healthy" if (speech_recognizer and db_connected) else "degraded",
        model_loaded=speech_recognizer is not None,
        db_connected=db_connected,
        version="1.0.0",
        uptime_seconds=uptime,
        statistics=stats
    )


@app.get("/api/statistics", response_model=StatisticsResponse)
@limiter.limit("10/second")
async def get_statistics(request: Request):
    """
    Get system statistics.
    
    Returns processing metrics, success rates, and counts.
    """
    stats = database.get_statistics()
    
    return StatisticsResponse(
        total_entries=stats['total_entries'],
        total_products=stats['total_products'],
        average_confidence=stats['average_confidence'],
        entries_today=stats['entries_today'],
        total_failures=stats['total_failures'],
        success_rate=stats['success_rate']
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=ErrorCodes.INTERNAL_ERROR,
            message=str(exc.detail)
        ).model_dump()
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()}
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
