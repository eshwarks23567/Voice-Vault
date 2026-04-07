"""
Audio Preprocessing Module for Voice Inventory System

This module handles noise reduction, normalization, and bandpass filtering
for audio signals before they are sent to the Whisper speech recognition model.
Designed for noisy industrial/warehouse environments.
"""

import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfilt
from typing import Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Handles audio preprocessing for speech recognition in noisy environments.
    
    Designed to improve speech recognition accuracy by:
    1. Reducing background noise (factory sounds, machinery, etc.)
    2. Applying bandpass filter for human speech frequencies
    3. Normalizing audio levels for consistent input to Whisper
    
    Attributes:
        target_sample_rate (int): Standard sample rate for processing (16000 Hz for Whisper)
        target_db (float): Target peak dB level for normalization (-20 dB default)
        low_freq (int): Low cutoff frequency for bandpass filter (300 Hz)
        high_freq (int): High cutoff frequency for bandpass filter (3400 Hz)
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_db: float = -16.0,
        low_freq: int = 80,
        high_freq: int = 8000
    ):
        """
        Initialize the AudioPreprocessor.
        
        Args:
            target_sample_rate: Target sample rate for output audio (default 16000 Hz for Whisper)
            target_db: Target peak dB level for normalization (default -16 dB for clearer audio)
            low_freq: Low cutoff frequency for speech bandpass filter (default 80 Hz)
            high_freq: High cutoff frequency for speech bandpass filter (default 8000 Hz for full speech)
        """
        self.target_sample_rate = target_sample_rate
        self.target_db = target_db
        self.low_freq = low_freq
        self.high_freq = high_freq
        
        logger.info(
            f"AudioPreprocessor initialized: sample_rate={target_sample_rate}, "
            f"target_db={target_db}, bandpass={low_freq}-{high_freq} Hz"
        )
    
    def reduce_noise(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        stationary: bool = True,
        prop_decrease: float = 0.75
    ) -> np.ndarray:
        """
        Remove background noise from audio using spectral gating.
        
        Uses the noisereduce library to apply spectral gating, which is effective
        for removing stationary background noise like factory machinery, HVAC,
        and other constant ambient sounds.
        
        Args:
            audio_data: Input audio signal as numpy array (1D float array)
            sample_rate: Sample rate of the audio in Hz
            stationary: If True, assumes noise is stationary (constant background).
                       If False, uses non-stationary noise reduction (slower but
                       better for varying noise). Default True for industrial settings.
            prop_decrease: Proportion to reduce noise by (0.0 to 1.0). Higher values
                          remove more noise but may affect speech quality. Default 0.75.
        
        Returns:
            np.ndarray: Noise-reduced audio signal as numpy array
        
        Raises:
            ValueError: If audio_data is empty or has invalid dimensions
        
        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> clean_audio = preprocessor.reduce_noise(noisy_audio, 16000)
        """
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data is empty or None")
        
        if audio_data.ndim > 1:
            # Convert stereo to mono by averaging channels
            audio_data = np.mean(audio_data, axis=1)
            logger.info("Converted stereo audio to mono")
        
        try:
            # Apply spectral gating noise reduction with GENTLE settings to preserve speech
            reduced_audio = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                stationary=stationary,
                prop_decrease=0.5,  # Less aggressive - preserve more speech
                n_fft=2048,
                hop_length=512,
                n_std_thresh_stationary=1.8,  # Higher threshold = gentler noise removal
                freq_mask_smooth_hz=500,  # Smoother frequency transitions
                time_mask_smooth_ms=150,  # More temporal smoothing for better quality
                thresh_n_mult_nonstationary=3,  # Less aggressive for varying noise
                sigmoid_slope_nonstationary=5  # Gentler noise gate to preserve speech
            )
            
            logger.debug(
                f"Noise reduction applied: input_rms={np.sqrt(np.mean(audio_data**2)):.4f}, "
                f"output_rms={np.sqrt(np.mean(reduced_audio**2)):.4f}"
            )
            
            return reduced_audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            # Return original audio if noise reduction fails
            return audio_data.astype(np.float32)
    
    def normalize_audio(
        self,
        audio_data: np.ndarray,
        target_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize audio to a target peak dB level.
        
        Adjusts the audio amplitude so that the peak level matches the target dB,
        ensuring consistent input levels to the speech recognition model.
        Also prevents clipping by ensuring values stay within [-1.0, 1.0].
        
        Args:
            audio_data: Input audio signal as numpy array
            target_db: Target peak dB level. If None, uses instance default (-20 dB).
                      Common values: -20 dB (conservative), -12 dB (louder), -6 dB (loud)
        
        Returns:
            np.ndarray: Normalized audio signal with peak at target_db
        
        Raises:
            ValueError: If audio_data is empty
        
        Example:
            >>> preprocessor = AudioPreprocessor(target_db=-20.0)
            >>> normalized = preprocessor.normalize_audio(audio)
        """
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data is empty or None")
        
        target_db = target_db if target_db is not None else self.target_db
        
        # Find current peak amplitude
        current_peak = np.max(np.abs(audio_data))
        
        if current_peak < 1e-10:  # Nearly silent audio
            logger.warning("Audio is nearly silent, skipping normalization")
            return audio_data.astype(np.float32)
        
        # Calculate current peak in dB
        current_db = 20 * np.log10(current_peak)
        
        # Calculate gain needed
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized_audio = audio_data * gain_linear
        
        # Prevent clipping - ensure values are within [-1.0, 1.0]
        max_val = np.max(np.abs(normalized_audio))
        if max_val > 1.0:
            normalized_audio = normalized_audio / max_val
            logger.warning("Clipping prevented during normalization")
        
        logger.debug(
            f"Normalization: current_db={current_db:.2f}, target_db={target_db:.2f}, "
            f"gain_db={gain_db:.2f}"
        )
        
        return normalized_audio.astype(np.float32)
    
    def apply_bandpass_filter(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        low_freq: Optional[int] = None,
        high_freq: Optional[int] = None,
        order: int = 5
    ) -> np.ndarray:
        """
        Apply bandpass filter to isolate human speech frequencies.
        
        Uses a Butterworth filter to pass frequencies in the human speech range
        (typically 300-3400 Hz) while attenuating noise outside this range.
        This helps remove low-frequency rumble and high-frequency hiss.
        
        Args:
            audio_data: Input audio signal as numpy array
            sample_rate: Sample rate of the audio in Hz
            low_freq: Low cutoff frequency in Hz. Default 300 Hz (human speech fundamental)
            high_freq: High cutoff frequency in Hz. Default 3400 Hz (telephone quality)
            order: Filter order. Higher = sharper cutoff but more ringing. Default 5.
        
        Returns:
            np.ndarray: Bandpass filtered audio signal
        
        Raises:
            ValueError: If audio_data is empty or sample_rate is invalid
        
        Note:
            The default range (300-3400 Hz) is telephone quality. For higher fidelity,
            consider 80-8000 Hz, but this may let through more noise.
        
        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> filtered = preprocessor.apply_bandpass_filter(audio, 16000)
        """
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data is empty or None")
        
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        low_freq = low_freq if low_freq is not None else self.low_freq
        high_freq = high_freq if high_freq is not None else self.high_freq
        
        # Calculate Nyquist frequency
        nyquist = sample_rate / 2
        
        # Normalize frequencies
        low_normalized = low_freq / nyquist
        high_normalized = high_freq / nyquist
        
        # Ensure frequencies are valid (between 0 and 1, exclusive)
        low_normalized = max(0.001, min(low_normalized, 0.999))
        high_normalized = max(0.001, min(high_normalized, 0.999))
        
        if low_normalized >= high_normalized:
            logger.warning("Invalid frequency range, skipping bandpass filter")
            return audio_data.astype(np.float32)
        
        try:
            # Design Butterworth bandpass filter using second-order sections (more stable)
            sos = butter(
                order,
                [low_normalized, high_normalized],
                btype='band',
                output='sos'
            )
            
            # Apply filter using second-order sections
            filtered_audio = sosfilt(sos, audio_data)
            
            logger.debug(
                f"Bandpass filter applied: {low_freq}-{high_freq} Hz, order={order}"
            )
            
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Bandpass filter failed: {e}")
            return audio_data.astype(np.float32)
    
    def preprocess_pipeline(
        self,
        audio_file_path: str,
        apply_noise_reduction: bool = True,
        apply_bandpass: bool = True,
        apply_normalization: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Complete preprocessing pipeline for audio files.
        
        Loads an audio file and applies the full preprocessing chain:
        1. Load and resample to target sample rate
        2. Convert to mono if stereo
        3. Apply noise reduction (optional)
        4. Apply bandpass filter (optional)
        5. Normalize audio levels (optional)
        
        Args:
            audio_file_path: Path to the audio file (WAV, MP3, FLAC, etc.)
            apply_noise_reduction: Whether to apply noise reduction. Default True.
            apply_bandpass: Whether to apply bandpass filter. Default True.
            apply_normalization: Whether to normalize audio. Default True.
        
        Returns:
            Tuple containing:
                - np.ndarray: Preprocessed audio signal ready for Whisper
                - int: Sample rate of the output audio
        
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the file format is not supported or audio is corrupted
        
        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> audio, sr = preprocessor.preprocess_pipeline("recording.wav")
            >>> # audio is now ready for Whisper transcription
        """
        # Validate file exists
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        logger.info(f"Processing audio file: {audio_file_path}")
        
        try:
            # Load audio file using librosa
            # sr=None preserves original sample rate, then we resample
            audio_data, original_sr = librosa.load(
                audio_file_path,
                sr=self.target_sample_rate,  # Resample to target
                mono=True  # Convert to mono
            )
            
            logger.info(
                f"Loaded audio: duration={len(audio_data)/self.target_sample_rate:.2f}s, "
                f"sample_rate={self.target_sample_rate}"
            )
            
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")
        
        # Validate audio is not empty
        if len(audio_data) == 0:
            raise ValueError("Audio file is empty")
        
        # Check for corrupted audio (all zeros or NaN)
        if np.all(audio_data == 0):
            raise ValueError("Audio file contains only silence")
        
        if np.any(np.isnan(audio_data)):
            raise ValueError("Audio file contains corrupted data (NaN values)")
        
        # Step 1: Noise reduction
        if apply_noise_reduction:
            logger.info("Applying noise reduction...")
            audio_data = self.reduce_noise(audio_data, self.target_sample_rate)
        
        # Step 2: Bandpass filter
        if apply_bandpass:
            logger.info("Applying bandpass filter...")
            audio_data = self.apply_bandpass_filter(audio_data, self.target_sample_rate)
        
        # Step 3: Normalization
        if apply_normalization:
            logger.info("Applying normalization...")
            audio_data = self.normalize_audio(audio_data)
        
        logger.info("Audio preprocessing complete")
        
        return audio_data, self.target_sample_rate
    
    def calculate_snr(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray
    ) -> float:
        """
        Calculate the Signal-to-Noise Ratio improvement.
        
        Estimates the SNR improvement by comparing RMS levels before and after
        processing. This is useful for evaluating preprocessing effectiveness.
        
        Args:
            original_audio: Original noisy audio signal
            processed_audio: Processed audio signal after noise reduction
        
        Returns:
            float: Estimated SNR improvement in dB
        """
        original_rms = np.sqrt(np.mean(original_audio ** 2))
        processed_rms = np.sqrt(np.mean(processed_audio ** 2))
        
        if processed_rms < 1e-10:
            return 0.0
        
        # Calculate noise removed (difference in energy)
        noise_estimate = np.sqrt(np.mean((original_audio - processed_audio) ** 2))
        
        if noise_estimate < 1e-10:
            return float('inf')
        
        snr_improvement = 20 * np.log10(processed_rms / noise_estimate)
        
        return snr_improvement
    
    def save_preprocessed_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        output_path: str
    ) -> str:
        """
        Save preprocessed audio to a file.
        
        Args:
            audio_data: Audio signal to save
            sample_rate: Sample rate of the audio
            output_path: Path where the file should be saved
        
        Returns:
            str: Path to the saved file
        """
        import soundfile as sf
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio_data, sample_rate)
        
        logger.info(f"Saved preprocessed audio to: {output_path}")
        return output_path


# Convenience function for quick preprocessing
def preprocess_audio_file(
    audio_path: str,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Quick function to preprocess an audio file with default settings.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Optional path to save preprocessed audio
    
    Returns:
        Tuple of (preprocessed_audio, sample_rate)
    """
    preprocessor = AudioPreprocessor()
    audio, sr = preprocessor.preprocess_pipeline(audio_path)
    
    if output_path:
        preprocessor.save_preprocessed_audio(audio, sr, output_path)
    
    return audio, sr


if __name__ == "__main__":
    # Example usage and testing
    print("Audio Preprocessor Module")
    print("=" * 50)
    
    # Create instance
    preprocessor = AudioPreprocessor()
    
    # Test with sample audio if available
    test_audio_path = os.path.join(
        os.path.dirname(__file__),
        "test_audio",
        "sample.wav"
    )
    
    if os.path.exists(test_audio_path):
        audio, sr = preprocessor.preprocess_pipeline(test_audio_path)
        print(f"Processed audio: {len(audio)} samples at {sr} Hz")
        print(f"Duration: {len(audio)/sr:.2f} seconds")
    else:
        print(f"No test audio found at: {test_audio_path}")
        print("Place a sample.wav file in test_audio/ directory to test")
