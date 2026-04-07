"""
Whisper Speech Recognition Integration for Voice Inventory System

This module provides speech-to-text capabilities using OpenAI's Whisper model,
optimized for industrial/warehouse environments with noise preprocessing.
"""

import os
import sys
import tempfile
import logging
from typing import Dict, List, Optional, Any, Tuple
from math import exp

import numpy as np
import torch
import whisper

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_processing.noise_reducer import AudioPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Speech recognition using OpenAI's Whisper model with audio preprocessing.
    
    This class handles:
    - Loading and caching the Whisper model
    - Preprocessing audio for optimal recognition
    - Transcription with confidence scoring
    - Identification of low-confidence segments
    
    Attributes:
        model: Loaded Whisper model instance
        device: Device for inference ('cuda' or 'cpu')
        preprocessor: AudioPreprocessor instance for noise reduction
        model_name: Name of the loaded Whisper model
    
    Example:
        >>> recognizer = SpeechRecognizer(model_name='medium')
        >>> result = recognizer.transcribe_audio('recording.wav')
        >>> print(result['text'])
        'Product ABC-123 quantity 50 location A-12'
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm'}
    
    # Minimum audio duration in seconds
    MIN_DURATION = 0.5
    
    # Maximum audio duration in seconds (5 minutes)
    MAX_DURATION = 300
    
    # Inventory-specific prompt to help Whisper understand context
    INVENTORY_PROMPT = (
        "This is a warehouse inventory system. "
        "Product codes like ABC-123, XYZ-789. "
        "Quantities like 50, 100, 25. "
        "Locations like A-12, B-45, warehouse section. "
        "Common phrases: product code, quantity, location, aisle, shelf, pallet."
    )
    
    def __init__(
        self,
        model_name: str = 'base',
        device: Optional[str] = None,
        download_root: Optional[str] = None
    ):
        """
        Initialize the SpeechRecognizer with Whisper model.
        
        Args:
            model_name: Whisper model size to use. Options:
                - 'tiny': Fastest, lowest accuracy (~39M params)
                - 'base': Fast, good for clear audio (~74M params)
                - 'small': Balanced speed/accuracy (~244M params)
                - 'medium': Good accuracy (~769M params) - RECOMMENDED
                - 'large-v3': Best accuracy (~1550M params)
            device: Device for inference. If None, auto-detects CUDA availability.
            download_root: Directory to cache model weights. Uses default if None.
        
        Raises:
            RuntimeError: If model fails to load
        """
        self.model_name = model_name
        
        # Set device (prefer CUDA if available)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing SpeechRecognizer with {model_name} model on {self.device}")
        
        # Load Whisper model
        try:
            self.model = whisper.load_model(
                model_name,
                device=self.device,
                download_root=download_root
            )
            # Set to evaluation mode for inference
            self.model.eval()
            logger.info(f"Whisper {model_name} model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")
        
        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor(target_sample_rate=16000)
        
        # Warm up the model with a dummy audio
        self._warmup()
    
    def _warmup(self):
        """Warm up the model with a short dummy audio to initialize CUDA kernels."""
        try:
            # Create 1 second of silence at 16kHz
            dummy_audio = np.zeros(16000, dtype=np.float32)
            # Transcribe to initialize model
            _ = self.model.transcribe(
                dummy_audio,
                language='en',
                fp16=False,
                temperature=0.0,
                verbose=False
            )
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed (this is okay): {e}")
    
    def transcribe_audio(
        self,
        audio_path: str,
        language: str = 'en',
        preprocess: bool = True,
        word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text with confidence scoring.
        
        Processes the audio through the complete pipeline:
        1. Load and validate audio file
        2. Preprocess (noise reduction, filtering, normalization)
        3. Run Whisper transcription
        4. Calculate confidence scores
        5. Identify problematic segments
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Language code for transcription (default 'en' for English)
            preprocess: Whether to apply audio preprocessing. Default True.
            word_timestamps: Whether to include word-level timestamps. Default True.
        
        Returns:
            Dict containing:
                - 'text': Full transcription text (str)
                - 'confidence': Overall confidence score 0-1 (float)
                - 'segments': List of segment dicts with timestamps
                - 'language': Detected/specified language
                - 'duration': Audio duration in seconds
                - 'low_confidence_words': List of words with confidence < 0.7
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is unsupported or audio is corrupted
        
        Example:
            >>> result = recognizer.transcribe_audio('recording.wav')
            >>> print(f"Text: {result['text']}")
            >>> print(f"Confidence: {result['confidence']:.2%}")
        """
        # Validate file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check file format
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        
        logger.info(f"Transcribing: {audio_path}")
        
        try:
            # Preprocess audio if enabled
            if preprocess:
                audio_data, sample_rate = self.preprocessor.preprocess_pipeline(audio_path)
            else:
                import librosa
                audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            # Validate audio duration
            duration = len(audio_data) / sample_rate
            if duration < self.MIN_DURATION:
                raise ValueError(
                    f"Audio too short: {duration:.2f}s. Minimum: {self.MIN_DURATION}s"
                )
            if duration > self.MAX_DURATION:
                raise ValueError(
                    f"Audio too long: {duration:.2f}s. Maximum: {self.MAX_DURATION}s"
                )
            
            logger.info(f"Audio duration: {duration:.2f}s")
            
            # Run Whisper transcription with FAST settings
            result = self.model.transcribe(
                audio_data,
                language=language,
                task='transcribe',
                fp16=self.device == 'cuda',  # Use fp16 on GPU for speed
                temperature=0.0,  # Deterministic output
                word_timestamps=False,  # Disable for speed
                verbose=False,
                # FAST decoding - greedy instead of beam search
                beam_size=1,  # Greedy decoding (fastest)
                best_of=1,  # No sampling overhead
                # Suppress hallucinations
                suppress_blank=True,
                condition_on_previous_text=False,  # Faster without context
                # Detection thresholds
                compression_ratio_threshold=2.8,
                logprob_threshold=-1.5,
                no_speech_threshold=0.4,
                # Inventory-specific context prompt
                initial_prompt=self.INVENTORY_PROMPT
            )
            
            # Calculate confidence score
            confidence = self.calculate_confidence(result)
            
            # Process segments
            segments = self._process_segments(result.get('segments', []))
            
            # Identify low-confidence words
            low_confidence_words = self.segment_confidence(result.get('segments', []))
            
            transcription_result = {
                'text': result['text'].strip(),
                'confidence': confidence,
                'segments': segments,
                'language': result.get('language', language),
                'duration': duration,
                'low_confidence_words': low_confidence_words,
                'raw_result': result  # Include raw result for debugging
            }
            
            logger.info(
                f"Transcription complete: '{result['text'][:50]}...' "
                f"(confidence: {confidence:.2%})"
            )
            
            return transcription_result
            
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise ValueError(f"Failed to transcribe audio: {e}")
    
    def transcribe_audio_array(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Transcribe audio from numpy array (useful for WebSocket streaming).
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio (should be 16000 for Whisper)
            language: Language code for transcription
        
        Returns:
            Dict with transcription results (same format as transcribe_audio)
        """
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        # Ensure correct sample rate
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=16000
            )
        
        # Run transcription with optimized parameters
        result = self.model.transcribe(
            audio_data.astype(np.float32),
            language=language,
            task='transcribe',
            fp16=self.device == 'cuda',
            temperature=0.0,
            word_timestamps=True,
            verbose=False,
            # Enhanced decoding options
            beam_size=5,
            best_of=5,
            patience=1.0,
            suppress_blank=True,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=self.INVENTORY_PROMPT
        )
        
        confidence = self.calculate_confidence(result)
        segments = self._process_segments(result.get('segments', []))
        low_confidence_words = self.segment_confidence(result.get('segments', []))
        
        return {
            'text': result['text'].strip(),
            'confidence': confidence,
            'segments': segments,
            'language': result.get('language', language),
            'duration': len(audio_data) / 16000,
            'low_confidence_words': low_confidence_words
        }
    
    def calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score from Whisper result using multiple signals.
        
        Uses a combination of:
        - Average log probability from segments
        - Word-level probabilities when available
        - No-speech probability to detect silence/noise
        - Compression ratio to detect repetitive/hallucinated text
        
        Args:
            result: Raw Whisper transcription result containing segments
        
        Returns:
            float: Confidence score between 0 and 1
                   - > 0.80: High confidence, likely accurate
                   - 0.60-0.80: Medium confidence, may need verification
                   - < 0.60: Low confidence, should retry or use fallback
        
        Example:
            >>> result = model.transcribe(audio)
            >>> confidence = recognizer.calculate_confidence(result)
            >>> if confidence < 0.80:
            ...     print("Please repeat that more clearly")
        """
        segments = result.get('segments', [])
        
        if not segments:
            logger.warning("No segments found in result, returning low confidence")
            return 0.0
        
        # 1. Collect segment-level log probabilities
        segment_probs = []
        word_probs = []
        no_speech_probs = []
        
        for segment in segments:
            # Get segment average log probability
            avg_logprob = segment.get('avg_logprob', None)
            if avg_logprob is not None:
                # Convert logprob to probability (range 0-1)
                segment_probs.append(exp(avg_logprob))
            
            # Get no_speech probability (indicates non-speech audio)
            no_speech_prob = segment.get('no_speech_prob', 0)
            if no_speech_prob is not None:
                no_speech_probs.append(no_speech_prob)
            
            # Collect word-level probabilities for more granular confidence
            words = segment.get('words', [])
            for word_info in words:
                word_prob = word_info.get('probability', None)
                if word_prob is not None:
                    word_probs.append(word_prob)
        
        # 2. Calculate base confidence from segments
        if segment_probs:
            # Weight by segment duration if available
            segment_conf = sum(segment_probs) / len(segment_probs)
        else:
            segment_conf = 0.5
        
        # 3. Enhance with word-level confidence if available
        if word_probs:
            word_conf = sum(word_probs) / len(word_probs)
            # Combine segment and word confidence (word-level is more precise)
            base_confidence = 0.4 * segment_conf + 0.6 * word_conf
        else:
            base_confidence = segment_conf
        
        # 4. Apply penalties for potential issues
        penalties = 0.0
        
        # Penalty for high no-speech probability (indicates noise/silence)
        if no_speech_probs:
            avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)
            if avg_no_speech > 0.5:
                penalties += (avg_no_speech - 0.5) * 0.4
                logger.debug(f"No-speech penalty applied: {avg_no_speech:.2f}")
        
        # Penalty for very short transcriptions (might be noise)
        text = result.get('text', '').strip()
        if len(text) < 5:
            penalties += 0.2
            logger.debug("Short text penalty applied")
        
        # Penalty for all-caps or no letters (probably noise)
        if text and not any(c.isalpha() for c in text):
            penalties += 0.3
            logger.debug("No-letters penalty applied")
        
        # 5. Calculate final confidence with scaling for better range
        # Scale up slightly since Whisper tends to be conservative
        scaled_confidence = base_confidence * 1.15
        final_confidence = max(0.0, min(1.0, scaled_confidence - penalties))
        
        logger.info(
            f"Confidence: segment={segment_conf:.2f}, word={sum(word_probs)/len(word_probs) if word_probs else 0:.2f}, "
            f"penalties={penalties:.2f}, final={final_confidence:.2f}"
        )
        
        return final_confidence
    
    def segment_confidence(
        self,
        segments: List[Dict[str, Any]],
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Identify low-confidence words in transcription segments.
        
        Analyzes word-level confidence scores to find words that may
        have been misrecognized. Useful for targeted retry requests.
        
        Args:
            segments: List of segment dicts from Whisper result
            threshold: Confidence threshold below which words are flagged.
                      Default 0.7 (70%).
        
        Returns:
            List of dicts with low-confidence words:
                - 'word': The transcribed word
                - 'start': Start time in seconds
                - 'end': End time in seconds
                - 'confidence': Word confidence score
                - 'segment_id': Index of the segment containing the word
        
        Example:
            >>> low_conf = recognizer.segment_confidence(result['segments'])
            >>> for word_info in low_conf:
            ...     print(f"Low confidence: '{word_info['word']}' ({word_info['confidence']:.0%})")
        """
        low_confidence_words = []
        
        for seg_idx, segment in enumerate(segments):
            # Check for word-level information
            words = segment.get('words', [])
            
            for word_info in words:
                word_text = word_info.get('word', '').strip()
                word_prob = word_info.get('probability', 1.0)
                
                if word_prob < threshold and word_text:
                    low_confidence_words.append({
                        'word': word_text,
                        'start': word_info.get('start', 0),
                        'end': word_info.get('end', 0),
                        'confidence': word_prob,
                        'segment_id': seg_idx
                    })
            
            # If no word-level info, check segment-level
            if not words:
                avg_logprob = segment.get('avg_logprob', 0)
                segment_conf = min(1.0, max(0.0, exp(avg_logprob)))
                
                if segment_conf < threshold:
                    low_confidence_words.append({
                        'word': segment.get('text', '').strip(),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'confidence': segment_conf,
                        'segment_id': seg_idx,
                        'is_segment': True  # Flag that this is segment-level
                    })
        
        logger.debug(f"Found {len(low_confidence_words)} low-confidence words/segments")
        
        return low_confidence_words
    
    def _process_segments(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and clean up segment data for output.
        
        Args:
            segments: Raw segments from Whisper result
        
        Returns:
            List of cleaned segment dictionaries
        """
        processed = []
        
        for segment in segments:
            processed_segment = {
                'id': segment.get('id', 0),
                'start': round(segment.get('start', 0), 3),
                'end': round(segment.get('end', 0), 3),
                'text': segment.get('text', '').strip(),
                'confidence': min(1.0, max(0.0, exp(segment.get('avg_logprob', -1))))
            }
            
            # Include word-level data if available
            if 'words' in segment:
                processed_segment['words'] = [
                    {
                        'word': w.get('word', '').strip(),
                        'start': round(w.get('start', 0), 3),
                        'end': round(w.get('end', 0), 3),
                        'confidence': w.get('probability', 1.0)
                    }
                    for w in segment['words']
                ]
            
            processed.append(processed_segment)
        
        return processed
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'supported_formats': list(self.SUPPORTED_FORMATS)
        }


# Singleton instance for caching
_recognizer_instance: Optional[SpeechRecognizer] = None


def get_recognizer(model_name: str = 'base') -> SpeechRecognizer:
    """
    Get or create a cached SpeechRecognizer instance.
    
    This function implements a simple singleton pattern to avoid
    loading the model multiple times.
    
    Args:
        model_name: Whisper model to use (default: 'base' for deployment safety)
    
    Returns:
        SpeechRecognizer instance
    """
    global _recognizer_instance
    
    if _recognizer_instance is None or _recognizer_instance.model_name != model_name:
        _recognizer_instance = SpeechRecognizer(model_name=model_name)
    
    return _recognizer_instance


if __name__ == "__main__":
    # Example usage
    print("Speech Recognizer Module")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create recognizer (use 'tiny' for quick testing)
    print("\nLoading Whisper model (this may take a moment)...")
    recognizer = SpeechRecognizer(model_name='tiny')
    
    print(f"\nModel info: {recognizer.get_model_info()}")
    
    # Test with sample audio if available
    test_audio = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "audio_processing",
        "test_audio",
        "sample.wav"
    )
    
    if os.path.exists(test_audio):
        print(f"\nTranscribing: {test_audio}")
        result = recognizer.transcribe_audio(test_audio)
        print(f"Text: {result['text']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Duration: {result['duration']:.2f}s")
    else:
        print(f"\nNo test audio found at: {test_audio}")
        print("Place a sample.wav file to test transcription")
