# smarttranscriber/modules/transcriptionEngine.py

"""
Smart Transcriber Engine
Handles transcription processing with automatic optimization and fallback
Part 3 of 5
"""

import os
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import time
from datetime import datetime, timedelta
import wave
import contextlib
from queue import Queue
import threading
from threading import Event, Lock
import tempfile
import subprocess
import psutil
import whisper

# Import from previous parts
from modules.systemConfig import ConfigurationManager, SystemType
from modules.hardwareManager import HardwareManager, HardwareType

@dataclass
class TranscriptionConfig:
    """Configuration to support turbo model and latest Whisper features"""
    model_size: str = "base"
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    compression_ratio_threshold: Optional[float] = 2.4
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    batch_size: Optional[int] = None

    def validate(self):
        """Validate configuration settings"""
        valid_models = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large", "turbo"
        ]
        if self.model_size not in valid_models:
            raise ValueError(f"Model size must be one of {valid_models}")

@dataclass
class TranscriptionStats:
    start_time: float
    end_time: float = 0.0
    audio_duration: float = 0.0
    processing_duration: float = 0.0
    peak_memory: float = 0.0
    peak_gpu_memory: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    model_size: str = ""
    device_used: str = ""
    success: bool = False
    error: Optional[str] = None

class ModelManager:
    def __init__(self, config_manager: ConfigurationManager,
                 hardware_manager: HardwareManager):
        self.config_manager = config_manager
        self.hardware_manager = hardware_manager
        self.logger = config_manager.logger
        self.loaded_models: Dict[str, whisper.Whisper] = {}
        self.model_locks: Dict[str, Lock] = {}
        self.model_cache_dir = 'models'  # Define model cache directory

        # Updated model requirements
        self.model_requirements = {
            "tiny": {"vram_gb": 1, "recommended_batch": 16},
            "tiny.en": {"vram_gb": 1, "recommended_batch": 16},
            "base": {"vram_gb": 1, "recommended_batch": 16},
            "base.en": {"vram_gb": 1, "recommended_batch": 16},
            "small": {"vram_gb": 2, "recommended_batch": 8},
            "small.en": {"vram_gb": 2, "recommended_batch": 8},
            "medium": {"vram_gb": 5, "recommended_batch": 4},
            "medium.en": {"vram_gb": 5, "recommended_batch": 4},
            "large": {"vram_gb": 10, "recommended_batch": 2},
            "turbo": {"vram_gb": 6, "recommended_batch": 8}
        }

    def _get_available_vram(self, device: str) -> float:
        """Get available VRAM for the specified device."""
        if device == 'cuda':
            # Get available VRAM for CUDA device
            if torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
                gpu_properties = torch.cuda.get_device_properties(gpu_id)
                total_vram = gpu_properties.total_memory
                reserved_vram = torch.cuda.memory_reserved(gpu_id)
                allocated_vram = torch.cuda.memory_allocated(gpu_id)
                free_vram = total_vram - reserved_vram - allocated_vram
                return free_vram
        # For other devices, return 0 or appropriate value
        return 0.0

    def suggest_optimal_model(self, device: str) -> str:
        """Suggest optimal model based on available hardware"""
        available_vram = 0

        if device == "cuda":
            # Get available VRAM for NVIDIA GPU
            stats = self.hardware_manager.current_stats.get(HardwareType.NVIDIA_GPU)
            if stats:
                available_vram = (stats.memory_total - stats.memory_used) / (1024**3)  # Convert to GB
        elif device == "mps":
            # For Apple Silicon
            stats = self.hardware_manager.current_stats.get(HardwareType.APPLE_SILICON)
            if stats:
                available_vram = (stats.memory_total - stats.memory_used) / (1024**3)

        # Suggest model based on available VRAM
        if available_vram >= 10:
            return "large"
        elif available_vram >= 6:
            return "turbo"  # Prefer turbo over medium if enough VRAM
        elif available_vram >= 5:
            return "medium"
        elif available_vram >= 2:
            return "small"
        else:
            return "base"

    def get_model(self, model_size: str, device: str) -> whisper.Whisper:
        """Get or load a model with specified size and device"""
        model_key = f"{model_size}_{device}"

        if model_key not in self.model_locks:
            self.model_locks[model_key] = Lock()

        with self.model_locks[model_key]:
            if model_key not in self.loaded_models:
                # Verify hardware requirements
                requirements = self.model_requirements[model_size]
                if device != "cpu":
                    available_vram = self._get_available_vram(device)
                    if available_vram < requirements["vram_gb"] * 1024**3:  # Convert GB to bytes
                        self.logger.warning(
                            f"Insufficient VRAM for {model_size} model "
                            f"(requires {requirements['vram_gb']}GB, "
                            f"available: {available_vram/(1024**3):.1f}GB). "
                            "Consider using a smaller model or CPU device."
                        )

                self.logger.info(f"Loading {model_size} model on {device}")
                try:
                    model = whisper.load_model(model_size, device=device,
                                            download_root=self.model_cache_dir)
                    self.loaded_models[model_key] = model
                except Exception as e:
                    self.logger.error(f"Error loading model: {e}")
                    raise

            return self.loaded_models[model_key]

    def unload_model(self, model_size: str, device: str):
        """Unload a specific model from memory"""
        model_key = f"{model_size}_{device}"
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            self.logger.info(f"Unloaded model {model_size} from {device}")

class AudioProcessor:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = config_manager.logger
        self._validate_ffmpeg()

    def _validate_ffmpeg(self):
        """Validate FFmpeg installation"""
        if not self.config_manager.check_ffmpeg():
            self.logger.warning("FFmpeg not found. Audio preprocessing may be limited.")

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file"""
        try:
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except:
            # Try FFmpeg for non-WAV files
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'error', '-show_entries',
                    'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                    audio_path
                ], capture_output=True, text=True)
                return float(result.stdout)
            except:
                self.logger.error(f"Could not determine duration of {audio_path}")
                return 0.0

    def preprocess_audio(self, input_path: str) -> str:
        """Preprocess audio file for optimal transcription"""
        try:
            output_path = str(Path(tempfile.gettempdir()) /
                            f"preprocessed_{Path(input_path).stem}.wav")

            # Convert to WAV with standard parameters
            subprocess.run([
                'ffmpeg', '-y', '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ac', '1',  # Mono
                '-ar', '16000',  # 16kHz
                output_path
            ], check=True, capture_output=True)

            self.logger.info(f"Preprocessed audio saved to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            return input_path

class TranscriptionEngine:
    def _get_huggingface_token(self) -> Optional[str]:
        """Get Hugging Face token from various sources"""
        # Try environment variables first
        token = (
            os.getenv('HUGGINGFACE_TOKEN') or
            os.getenv('HF_TOKEN') or
            os.getenv('HUGGING_FACE_TOKEN')
        )

        # If not in env vars, try the local token file
        if not token:
            token_path = os.path.expanduser('~/.huggingface/token')
            try:
                if os.path.exists(token_path):
                    with open(token_path, 'r') as f:
                        token = f.read().strip()
                    self.logger.info("Found Hugging Face token in local configuration")
            except Exception as e:
                self.logger.debug(f"Error reading local token file: {e}")

        return token

    def __init__(self, config_manager: ConfigurationManager,
                 hardware_manager: HardwareManager,
                 performance_analyzer: 'PerformanceAnalyzer'):
        self.config_manager = config_manager
        self.logger = config_manager.logger
        self.hardware_manager = hardware_manager
        self.performance_analyzer = performance_analyzer
        self.model_manager = ModelManager(config_manager, hardware_manager)
        self.audio_processor = AudioProcessor(config_manager)
        self.process_monitor = None  # Will be set during initialization
        self.stop_event = Event()

        # Initialize diarization pipeline
        self.diarization_pipeline = None
        try:
            # Get Hugging Face token
            hf_token = self._get_huggingface_token()

            if not hf_token:
                self.logger.warning("No Hugging Face token found in environment or local config.")
            else:
                self.logger.info("Hugging Face token loaded successfully")

            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=hf_token
            )

            # Use GPU if available
            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
                self.logger.info("Using GPU for speaker diarization")
            else:
                self.logger.info("Using CPU for speaker diarization")

        except Exception as e:
            self.logger.error(f"Failed to initialize diarization pipeline: {e}")
            if "Unauthorized" in str(e):
                self.logger.error("Invalid or missing Hugging Face token. Please check your .env file or ~/.huggingface/token")

    def transcribe_file(self, audio_path: str, config: TranscriptionConfig = None) -> Dict:
        """Transcribe a single audio file with speaker diarization"""
        if config is None:
            config = TranscriptionConfig()
        config.validate()

        self.current_stats = TranscriptionStats(start_time=time.time())
        self.current_audio_path = audio_path

        try:
            # Preprocess audio
            self.logger.info(f"Processing: {audio_path}")
            preprocessed_path = self.audio_processor.preprocess_audio(audio_path)

            # Update progress to 10% after preprocessing
            self.process_monitor.update_progress(str(audio_path), 0.1)

            # Get optimal device and model
            device, device_info = self.hardware_manager.get_optimal_device()
            self.logger.info(f"Selected device: {device} ({device_info['name']})")

            # Perform diarization first
            self.logger.info("Starting speaker diarization...")
            speaker_segments = self._perform_diarization(preprocessed_path)

            # Update progress to 30% after diarization
            self.process_monitor.update_progress(str(audio_path), 0.3)

            # Perform transcription
            try:
                result = self._transcribe_with_device(
                    preprocessed_path, device, config
                )
                # Update progress to 90% after transcription
                self.process_monitor.update_progress(str(audio_path), 0.9)

            except Exception as e:
                self.logger.warning(f"Transcription failed on {device}: {e}")
                if device != "cpu":
                    self.logger.info("Falling back to CPU transcription")
                    device = "cpu"
                    result = self._transcribe_with_device(
                        preprocessed_path, device, config
                    )
                else:
                    raise

            # Add speaker information to segments
            result = self._align_transcription_with_speakers(result, speaker_segments)

            # Update statistics
            self.current_stats.end_time = time.time()
            self.current_stats.audio_duration = self.audio_processor.get_audio_duration(audio_path)
            self.current_stats.processing_duration = (
                self.current_stats.end_time - self.current_stats.start_time
            )
            self.current_stats.device_used = device
            self.current_stats.success = True
            self.current_stats.model_size = config.model_size

            # Add processing statistics to result
            result['processing_stats'] = asdict(self.current_stats)

            # Pass stats to the PerformanceAnalyzer
            self.performance_analyzer.add_stats(asdict(self.current_stats))

            # Final progress update
            self.process_monitor.update_progress(str(audio_path), 1.0)

            return result

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            self.current_stats.error = str(e)
            raise

    def set_process_monitor(self, monitor):
        """Set the process monitor after initialization"""
        self.process_monitor = monitor

    def _perform_diarization(self, audio_path: str) -> Dict[str, List[Tuple[float, float]]]:
        """Perform speaker diarization on audio file"""
        if not self.diarization_pipeline:
            self.logger.warning("Diarization pipeline not available")
            return {}

        try:
            # Perform diarization
            self.logger.info("Running speaker diarization...")
            diarization = self.diarization_pipeline(audio_path)

            # Organize speaker segments
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((turn.start, turn.end))

            self.logger.info(f"Identified {len(speaker_segments)} speakers")
            return speaker_segments

        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return {}

    def _align_transcription_with_speakers(self, result: Dict, speaker_segments: Dict[str, List[Tuple[float, float]]]) -> Dict:
            """Align transcription segments with speaker segments and detect gender"""
            # Initialize speaker characteristics
            speaker_characteristics = {}

            # First pass: analyze speaker segments for gender detection
            for speaker, turns in speaker_segments.items():
                total_duration = sum(end - start for start, end in turns)
                avg_duration = total_duration / len(turns) if turns else 0
                short_segments = sum(1 for start, end in turns if (end - start) < 2.0)
                long_segments = len(turns) - short_segments

                # Assign gender based on segment patterns
                is_female = (short_segments > long_segments) or (avg_duration < 2.0)
                speaker_characteristics[speaker] = {
                    'gender': 'female' if is_female else 'male',
                    'total_duration': total_duration,
                    'avg_duration': avg_duration,
                    'short_segments': short_segments,
                    'long_segments': long_segments
                }

            # Second pass: assign segments to speakers
            for segment in result["segments"]:
                segment_mid = (segment["start"] + segment["end"]) / 2
                segment_duration = segment["end"] - segment["start"]

                # Find the most likely speaker for this segment
                best_speaker = None
                best_overlap = 0

                for speaker, turns in speaker_segments.items():
                    for start, end in turns:
                        if start <= segment_mid <= end:
                            overlap = min(end, segment["end"]) - max(start, segment["start"])
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_speaker = speaker

                if best_speaker:
                    # Add both speaker ID and gender to segment
                    gender = speaker_characteristics[best_speaker]['gender']
                    segment["speaker"] = best_speaker
                    segment["gender"] = gender
                    segment["speaker_type"] = f"{gender}_speaker"
                else:
                    # Fallback to duration-based gender detection
                    gender = 'female' if segment_duration < 2.0 else 'male'
                    segment["speaker"] = f"unknown_{gender}"
                    segment["gender"] = gender
                    segment["speaker_type"] = f"{gender}_speaker"

            # Add speaker analysis to result
            result["speaker_analysis"] = speaker_characteristics
            return result

    def _transcribe_with_device(self, audio_path: str, device: str, config: TranscriptionConfig) -> Dict:
        model = self.model_manager.get_model(config.model_size, device)
        # Reset peak memory stats
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Prepare transcription options
        options = {
            "task": config.task,
            "language": config.language,
            "beam_size": config.beam_size,
            "best_of": config.best_of,
            "temperature": config.temperature,
            "compression_ratio_threshold": config.compression_ratio_threshold,
            "condition_on_previous_text": config.condition_on_previous_text,
            "initial_prompt": config.initial_prompt,
            "verbose": True,
            "word_timestamps": True
        }

        # Monitor memory usage during transcription
        def memory_monitor():
            process = psutil.Process()
            while not self.stop_event.is_set():
                # CPU usage
                cpu_percent = process.cpu_percent(interval=0.1)
                self.current_stats.cpu_usage = max(self.current_stats.cpu_usage, cpu_percent)

                # Memory usage
                memory_used = process.memory_info().rss / (1024**3)
                self.current_stats.peak_memory = max(self.current_stats.peak_memory, memory_used)

                # GPU memory usage
                if device == "cuda" and torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                    self.current_stats.peak_gpu_memory = max(self.current_stats.peak_gpu_memory, gpu_memory_used)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        self.current_stats.gpu_usage = max(self.current_stats.gpu_usage, gpu_util)
                    except:
                        pass
                time.sleep(0.1)

        # Start memory monitoring
        self.stop_event.clear()
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()

        try:
            # Perform transcription
            result = model.transcribe(audio_path, **options)
            # Add segments with improved timestamps
            result["segments"] = self._refine_segments(result["segments"])
            return result
        finally:
            # Stop memory monitoring
            self.stop_event.set()
            monitor_thread.join()

    def _refine_segments(self, segments: List[Dict]) -> List[Dict]:
        """Refine segment timestamps and add confidence scores"""
        refined_segments = []
        total_segments = len(segments)

        for idx, segment in enumerate(segments):
            refined_segment = segment.copy()

            # Adjust timestamps for better subtitle synchronization
            if idx > 0:
                prev_end = refined_segments[idx-1]["end"]
                if segment["start"] < prev_end:
                    refined_segment["start"] = prev_end

            # Add confidence score if available
            if "avg_logprob" in segment:
                refined_segment["confidence"] = self._logprob_to_confidence(
                    segment["avg_logprob"]
                )

            refined_segments.append(refined_segment)

        return refined_segments

    @staticmethod
    def _logprob_to_confidence(logprob: float) -> float:
        """Convert log probability to confidence score (0-1)"""
        return max(0.0, min(1.0, 1.0 + logprob))

    def cleanup(self):
        """Cleanup resources and temporary files"""
        # Unload all models
        for model_size in ["tiny", "base", "small", "medium", "large", "turbo"]:
            for device in ["cuda", "cpu", "mps"]:
                try:
                    self.model_manager.unload_model(model_size, device)
                except:
                    pass

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean temporary files
        try:
            for file in Path(tempfile.gettempdir()).glob("preprocessed_*.wav"):
                file.unlink()
        except Exception as e:
            self.logger.warning(f"Error cleaning temporary files: {e}")


    def _save_outputs(self, result: Dict, output_dir: Path, base_name: str):
        """Save output files with improved speaker separation"""
        # Create speaker segments dictionary
        speaker_segments = {
            'male_speaker': [],
            'female_speaker': []
        }

        # Organize segments by speaker
        for segment in result["segments"]:
            speaker = segment.get("speaker_type", "unknown_speaker")
            if speaker in speaker_segments:
                speaker_segments[speaker].append(segment)
            else:
                # Fallback for unknown speakers
                speaker_segments['male_speaker' if len(speaker_segments['male_speaker']) < len(speaker_segments['female_speaker']) else 'female_speaker'].append(segment)

        # Save separate SRT files for each speaker
        for speaker, segments in speaker_segments.items():
            if segments:  # Only create file if speaker has segments
                srt_path = output_dir / f"{base_name}_{speaker}.srt"
                self.generate_srt(segments, str(srt_path))
                self.logger.info(f"Generated SRT for {speaker}: {srt_path}")

                # Also save speaker-specific text file
                txt_path = output_dir / f"{base_name}_{speaker}.txt"
                text_content = "\n".join(segment["text"] for segment in segments)
                txt_path.write_text(text_content, encoding='utf-8')

        # Save combined plain text
        txt_path = output_dir / f"{base_name}.txt"
        txt_path.write_text(result["text"], encoding='utf-8')

        # Save JSON with detailed information
        json_path = output_dir / f"{base_name}.json"
        # Add speaker statistics
        result["speaker_statistics"] = {
            speaker: {
                "segment_count": len(segments),
                "total_duration": sum(s["end"] - s["start"] for s in segments),
                "average_segment_length": sum(s["end"] - s["start"] for s in segments) / len(segments) if segments else 0
            }
            for speaker, segments in speaker_segments.items()
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Log the speaker statistics
        for speaker, stats in result["speaker_statistics"].items():
            self.logger.info(
                f"{speaker}: {stats['segment_count']} segments, "
                f"{stats['total_duration']:.1f}s total duration, "
                f"{stats['average_segment_length']:.1f}s avg length"
            )

    def generate_srt(self, segments: List[Dict], output_path: str):
        """Generate SRT subtitle file from segments"""
        def format_timestamp(seconds: float) -> str:
            td = timedelta(seconds=seconds)
            hours = td.seconds // 3600
            minutes = (td.seconds % 3600) // 60
            seconds = td.seconds % 60
            milliseconds = td.microseconds // 1000
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                text = segment['text'].strip()

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
