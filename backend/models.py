"""
Pydantic Data Models for Voice Inventory System

Defines request/response schemas for API validation,
database models, and internal data structures.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ProcessingStatus(str, Enum):
    """Status codes for processing results."""
    SUCCESS = "success"
    LOW_CONFIDENCE = "low_confidence"
    INCOMPLETE = "incomplete"
    ERROR = "error"
    RETRY = "retry"
    PENDING = "pending"


class TranscriptionSegment(BaseModel):
    """Represents a segment of transcribed audio."""
    id: int
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    text: str
    confidence: float = Field(ge=0, le=1)
    words: Optional[List[Dict[str, Any]]] = None


class TranscriptionResult(BaseModel):
    """Result from speech-to-text processing."""
    text: str = Field(..., description="Full transcription text")
    confidence: float = Field(ge=0, le=1, description="Overall confidence score")
    language: str = Field(default="en")
    duration: float = Field(ge=0, description="Audio duration in seconds")
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    low_confidence_words: List[Dict[str, Any]] = Field(default_factory=list)


class InventoryEntryData(BaseModel):
    """Validated inventory entry data."""
    product_id: str = Field(..., pattern=r'^[A-Z]{2,3}-\d{3,5}$')
    quantity: int = Field(..., ge=1, le=9999)
    location: str = Field(..., pattern=r'^[A-Z]-\d{1,2}$')
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('product_id')
    @classmethod
    def uppercase_product_id(cls, v: str) -> str:
        return v.upper()
    
    @field_validator('location')
    @classmethod
    def uppercase_location(cls, v: str) -> str:
        return v.upper()


class ValidationResult(BaseModel):
    """Result from field validation."""
    status: ProcessingStatus
    data: Optional[InventoryEntryData] = None
    missing_fields: List[str] = Field(default_factory=list)
    found_fields: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    raw_text: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    retry: bool = False


class ProductInfo(BaseModel):
    """Product information from database."""
    id: int
    product_code: str
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class InventoryEntry(BaseModel):
    """Complete inventory entry with all metadata."""
    id: int
    product_code: str
    product_name: Optional[str] = None
    category: Optional[str] = None
    quantity: int
    location: str
    confidence_score: float
    raw_transcription: Optional[str] = None
    audio_path: Optional[str] = None
    verified: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None


class AudioUploadRequest(BaseModel):
    """Request for audio file upload."""
    language: str = Field(default="en")
    preprocess: bool = Field(default=True)


class AudioUploadResponse(BaseModel):
    """Response from audio file upload processing."""
    status: ProcessingStatus
    message: str
    transcription: Optional[TranscriptionResult] = None
    validation: Optional[ValidationResult] = None
    entry_id: Optional[int] = None
    processing_time_ms: float = 0


class ManualEntryRequest(BaseModel):
    """Request for manual entry validation."""
    product_id: str = Field(..., min_length=4)
    quantity: int = Field(..., ge=1, le=9999)
    location: str = Field(..., min_length=2)
    
    @field_validator('product_id')
    @classmethod
    def uppercase_product_id(cls, v: str) -> str:
        return v.upper()
    
    @field_validator('location')
    @classmethod
    def uppercase_location(cls, v: str) -> str:
        return v.upper()


class ManualEntryResponse(BaseModel):
    """Response from manual entry validation."""
    status: ProcessingStatus
    message: str
    entry_id: Optional[int] = None
    product_details: Optional[ProductInfo] = None


class InventoryQueryParams(BaseModel):
    """Query parameters for inventory search."""
    product_code: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class InventoryListResponse(BaseModel):
    """Response containing list of inventory entries."""
    entries: List[InventoryEntry]
    total_count: int
    limit: int
    offset: int
    has_more: bool


class ProductListResponse(BaseModel):
    """Response containing list of products."""
    products: List[ProductInfo]
    total_count: int


class HealthCheckResponse(BaseModel):
    """Health check endpoint response."""
    status: str = "healthy"
    model_loaded: bool = False
    db_connected: bool = False
    version: str = "1.0.0"
    uptime_seconds: float = 0
    statistics: Optional[Dict[str, Any]] = None


class WebSocketMessage(BaseModel):
    """Message format for WebSocket communication."""
    type: str = Field(..., description="Message type: 'audio', 'command', 'result'")
    data: Optional[Dict[str, Any]] = None
    audio_base64: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketResponse(BaseModel):
    """Response format for WebSocket messages."""
    status: ProcessingStatus
    message: str
    data: Optional[Dict[str, Any]] = None
    confidence: float = 0
    retry_field: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FallbackRequest(BaseModel):
    """Request to switch to fallback mode."""
    mode: str = Field(..., description="'digit_by_digit', 'slower', 'manual'")
    field_name: Optional[str] = None
    partial_value: Optional[str] = None


class FailureLog(BaseModel):
    """Log entry for failed attempts."""
    id: int
    audio_path: Optional[str]
    raw_transcription: Optional[str]
    error_type: str
    error_message: str
    confidence_score: Optional[float]
    attempt_count: int
    created_at: datetime


class StatisticsResponse(BaseModel):
    """Response containing system statistics."""
    total_entries: int
    total_products: int
    average_confidence: float
    entries_today: int
    total_failures: int
    success_rate: float
    processing_times: Optional[Dict[str, float]] = None


class ErrorResponse(BaseModel):
    """Standard error response format."""
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Error codes for consistent error handling
class ErrorCodes:
    """Standard error codes for the API."""
    AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MISSING_FIELDS = "MISSING_FIELDS"
    INVALID_PRODUCT = "INVALID_PRODUCT"
    INVALID_LOCATION = "INVALID_LOCATION"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    DATABASE_ERROR = "DATABASE_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    CONNECTION_TIMEOUT = "CONNECTION_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
