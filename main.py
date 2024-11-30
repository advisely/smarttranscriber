# smarttranscriber/main.py

"""
Smart Transcriber Main Application
Integrates all components and provides CLI interface
Part 5 of 5
"""

__version__ = "1.0.0"
__author__ = "Yassine Boumiza"
__license__ = "MIT"

# Standard library imports
import os
from dotenv import load_dotenv
import argparse
import json
import logging
import os
import sys
import signal
from pathlib import Path
import time
from typing import List, Dict, Optional, Tuple  # Make sure Tuple is here
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
from datetime import datetime

# Third-party imports
import uvicorn
import psutil

# Local module imports
from modules.systemConfig import ConfigurationManager, SystemType
from modules.hardwareManager import HardwareManager, HardwareType
from modules.transcriptionEngine import TranscriptionEngine, TranscriptionConfig
from modules.monitoring import ProcessMonitor, PerformanceAnalyzer
from modules.apiStructure import SmartTranscriberAPI

def display_startup_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   Smart Whisper Transcriber                   ║
║                      System Startup Check                     ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_system_requirements(config_manager: ConfigurationManager,
                            hardware_manager: HardwareManager) -> bool:
    print("\nPerforming system checks...")

    # Check required directories and disk space
    required_dirs = ['audio_files', 'output', 'models', 'data']
    required_space_gb = 10  # Require 10GB free space

    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"Creating {dir_name} directory...")
            dir_path.mkdir(parents=True, exist_ok=True)

        # Check available disk space
        try:
            free_space_gb = psutil.disk_usage(dir_path).free / (1024**3)
            if free_space_gb < required_space_gb:
                print(f"✗ Warning: Low disk space in {dir_name}: {free_space_gb:.1f}GB free")
                print(f"  Recommended minimum: {required_space_gb}GB")
                return False
            else:
                print(f"✓ {dir_name} directory: {free_space_gb:.1f}GB free")
        except Exception as e:
            print(f"✗ Error checking disk space for {dir_name}: {e}")
            return False

    # Check FFmpeg
    print("\n1. Checking FFmpeg installation...")
    if config_manager.check_ffmpeg():
        print("✓ FFmpeg is installed and accessible")
    else:
        print("✗ FFmpeg not found. Please install FFmpeg to process audio files")
        return False

    # Check Python version
    print("\n2. Checking Python compatibility...")
    if config_manager.check_python_compatibility():
        print(f"✓ Python version {sys.version.split()[0]} is compatible")
    else:
        print("✗ Python version is not compatible")
        return False

    # Check GPU capabilities
    print("\n3. Detecting hardware capabilities...")
    gpu_info = config_manager.get_gpu_capabilities()
    if gpu_info["cuda"]:
        print("✓ NVIDIA GPU detected and CUDA is available")
        for gpu in gpu_info["detected_gpus"]:
            print(f"  - {gpu['name']} ({gpu['memory']/(1024**3):.1f}GB VRAM)")
    elif gpu_info["rocm"]:
        print("✓ AMD GPU detected with ROCm support")
    elif gpu_info["mps"]:
        print("✓ Apple Silicon detected with MPS support")
    else:
        print("ℹ No GPU detected, will use CPU for processing")

    # Show system configuration
    print("\n4. Current system configuration:")
    print(f"✓ System type: {config_manager.system_type.name}")
    print(f"✓ Working directory: {os.getcwd()}")
    print(f"✓ Available CPU cores: {psutil.cpu_count()}")
    print(f"✓ Available RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")

    return True

def select_language() -> str:
    """Interactive language selection"""
    languages = {
        "1": ("en", "English (Default)"),
        "2": ("fr", "French"),
        "3": ("es", "Spanish"),
        "4": ("ar", "Arabic")
    }

    print("\nAvailable languages:")
    for key, (code, name) in languages.items():
        print(f"{key}. {name}")

    while True:
        choice = input("\nSelect language number [1]: ").strip()
        if choice == "":
            return "en"
        if choice in languages:
            return languages[choice][0]
        print("Invalid selection. Please try again.")

def select_workers(max_workers: int) -> int:
    """Interactive worker count selection"""
    print(f"\nWorker threads (1-{max_workers}, default: 1)")
    while True:
        choice = input(f"Enter number of workers [1]: ").strip()
        if choice == "":
            return 1
        try:
            workers = int(choice)
            if 1 <= workers <= max_workers:
                return workers
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {max_workers}")

def select_device(has_cuda: bool, has_mps: bool) -> str:
    """Interactive device selection"""
    devices = []
    if has_cuda:
        devices.append(("cuda", "NVIDIA GPU (Default)"))
    if has_mps:
        devices.append(("mps", "Apple Silicon"))
    devices.append(("cpu", "CPU" if has_cuda or has_mps else "CPU (Default)"))

    print("\nAvailable devices:")
    for i, (device, desc) in enumerate(devices, 1):
        print(f"{i}. {desc}")

    while True:
        choice = input("\nSelect device number [1]: ").strip()
        if choice == "":
            return devices[0][0]  # Return first device as default
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(devices):
                return devices[idx][0]
        except ValueError:
            pass
        print("Invalid selection. Please try again.")

def select_reports() -> bool:
    """Select whether to generate reports"""
    while True:
        choice = input("\nGenerate performance reports? (Y/n): ").lower().strip()
        if choice in ["", "y", "yes"]:
            return True
        if choice in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'")

def select_verbose() -> bool:
    """Select verbose output"""
    while True:
        choice = input("\nEnable verbose output? (Y/n): ").lower().strip()
        if choice in ["", "y", "yes"]:
            return True
        if choice in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'")

def get_max_workers() -> int:
    """Calculate maximum recommended workers based on system"""
    cpu_count = os.cpu_count() or 1
    available_ram_gb = psutil.virtual_memory().available / (1024**3)

    # Consider both CPU cores and available RAM
    # Assume each worker needs ~2GB RAM
    max_by_ram = int(available_ram_gb / 2)
    max_by_cpu = max(1, cpu_count - 1)  # Leave one core free

    return min(max_by_ram, max_by_cpu)

def load_env_variables():
    """Load environment variables from .env file"""
    # Load from .env file
    dotenv_path = Path('.env')
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

    # Set Hugging Face token if available (check multiple possible names)
    hf_token = (
        os.getenv('HUGGINGFACE_TOKEN') or
        os.getenv('HF_TOKEN') or
        os.getenv('HUGGING_FACE_TOKEN')
    )

    if hf_token:
        os.environ['HUGGINGFACE_TOKEN'] = hf_token
        print("Hugging Face token loaded successfully")
    else:
        print("Warning: No Hugging Face token found in environment")

class ApplicationConfig:
    def __init__(self):
        self.input_paths: List[Path] = []
        self.output_dir: Path = Path("output")
        self.model_size: str = "base"
        self.language: Optional[str] = None
        self.num_workers: int = 1
        self.batch_size: Optional[int] = None
        self.force_device: Optional[str] = None
        self.generate_reports: bool = True
        self.verbose: bool = False

def initialize_with_config() -> Tuple[bool, Optional[ApplicationConfig]]:
    """Initialize system and get user configuration"""
    load_env_variables()

def initialize_system(input_file: str = None) -> bool:
    display_startup_banner()

    config_manager = ConfigurationManager(audio_file=input_file)
    hardware_manager = HardwareManager(config_manager)

    if not check_system_requirements(config_manager, hardware_manager):
        return False

    print("\nAll system checks completed. Ready to start?")
    response = input("Continue? (y/n): ").lower().strip()

    return response == 'y'

def start_api_server():
    """Start the FastAPI server"""
    app = SmartTranscriberAPI().app
    uvicorn.run(app, host="0.0.0.0", port=8000)

class SmartTranscriber:
    def __init__(self, audio_file=None):
        # Initialize configuration manager first
        self.config_manager = ConfigurationManager(audio_file=audio_file)
        self.logger = self.config_manager.logger

        # Initialize other components
        self.hardware_manager = HardwareManager(self.config_manager)
        self.performance_analyzer = PerformanceAnalyzer(self.config_manager)

        # First create the transcription engine
        self.transcription_engine = TranscriptionEngine(
            self.config_manager,
            self.hardware_manager,
            self.performance_analyzer
        )

        # Then create process monitor
        self.process_monitor = ProcessMonitor(
            self.config_manager,
            self.hardware_manager,
            self.performance_analyzer
        )

        # Set the process monitor in the transcription engine
        self.transcription_engine.set_process_monitor(self.process_monitor)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.shutdown_event = threading.Event()
        self.logger.info("Smart Transcriber initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.shutdown_event.set()
        self.cleanup()
        sys.exit(0)

    def process_files(self, config: ApplicationConfig) -> Dict:
        """Process all input files according to configuration"""
        start_time = time.time()
        results = {}

        try:
            # Start monitoring
            self.hardware_manager.start_monitoring()
            self.process_monitor.start_monitoring()

            # Set total files in monitor
            total_files = len(config.input_paths)
            self.process_monitor.set_total_files(total_files)

            # Prepare transcription configuration
            transcription_config = TranscriptionConfig(
                model_size=config.model_size,
                language=config.language,
                batch_size=config.batch_size
            )

            # Process files
            if config.num_workers > 1:
                results = self._process_files_parallel(
                    config.input_paths,
                    transcription_config,
                    config.num_workers
                )
            else:
                results = self._process_files_sequential(
                    config.input_paths,
                    transcription_config
                )

            # Generate reports if requested
            if config.generate_reports:
                report_path = self.process_monitor.generate_final_report(
                    config.output_dir
                )
                self.logger.info(f"Processing report saved to: {report_path}")

            return results

        finally:
            self.cleanup()

    def _process_files_parallel(self, input_paths: List[Path],
                              transcription_config: TranscriptionConfig,
                              num_workers: int) -> Dict:
        """Process files in parallel using thread pool"""
        results = {}
        total_files = len(input_paths)
        processed_files = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_path = {
                executor.submit(
                    self._process_single_file,
                    input_path,
                    transcription_config
                ): input_path
                for input_path in input_paths
            }

            for future in as_completed(future_to_path):
                input_path = future_to_path[future]
                try:
                    result = future.result()
                    results[str(input_path)] = result
                    processed_files += 1

                    # Update progress
                    self.process_monitor.update_progress(
                        str(input_path),
                        processed_files / total_files
                    )

                except Exception as e:
                    self.logger.error(f"Error processing {input_path}: {e}")
                    results[str(input_path)] = {"error": str(e)}

                if self.shutdown_event.is_set():
                    break

        return results

    def _process_files_sequential(self, input_paths: List[Path],
                                transcription_config: TranscriptionConfig) -> Dict:
        """Process files sequentially"""
        results = {}
        total_files = len(input_paths)

        for i, input_path in enumerate(input_paths):
            if self.shutdown_event.is_set():
                break

            try:
                # Update progress before processing
                self.process_monitor.update_progress(
                    str(input_path),
                    0.0  # Start at 0% for this file
                )

                result = self._process_single_file(
                    input_path,
                    transcription_config
                )
                results[str(input_path)] = result

                # Update progress after completion
                self.process_monitor.update_progress(
                    str(input_path),
                    1.0  # 100% for this file
                )

            except Exception as e:
                self.logger.error(f"Error processing {input_path}: {e}")
                results[str(input_path)] = {"error": str(e)}

        return results

    def _process_single_file(self, input_path: Path,
                        transcription_config: TranscriptionConfig) -> Dict:
        """Process a single file and save outputs"""
        self.logger.info(f"Processing: {input_path}")

        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path("output") / f"{input_path.stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Transcribe file
        result = self.transcription_engine.transcribe_file(
            str(input_path),
            transcription_config
        )

        # Save outputs
        self._save_outputs(result, output_dir, input_path.stem)

        return result

    def _save_outputs(self, result: Dict, output_dir: Path, base_name: str):
        """Save output files with improved speaker separation"""
        # Create speaker segments dictionary
        speaker_segments = {
            'male_speaker': [],
            'female_speaker': []
        }

        # First pass: analyze segments for speaker characteristics
        for segment in result["segments"]:
            # Use speaker information from diarization
            if "speaker" in segment and "gender" in segment:
                speaker_key = f"{segment['gender']}_speaker"
                if speaker_key in speaker_segments:
                    speaker_segments[speaker_key].append(segment)
                else:
                    # Fallback if gender not detected
                    speaker_segments['male_speaker'].append(segment)
            else:
                # If no speaker/gender info, use duration-based heuristic
                duration = segment["end"] - segment["start"]
                if duration < 2.0:  # Short segments often female speech
                    speaker_segments['female_speaker'].append(segment)
                else:
                    speaker_segments['male_speaker'].append(segment)

        # Save separate SRT and TXT files for each speaker
        for speaker, segments in speaker_segments.items():
            if segments:  # Only create files if speaker has segments
                # Sort segments by start time
                segments.sort(key=lambda x: x["start"])

                # Generate SRT file
                srt_path = output_dir / f"{base_name}_{speaker}.srt"
                self.transcription_engine.generate_srt(segments, str(srt_path))
                self.logger.info(f"Generated SRT for {speaker}: {srt_path}")

                # Generate speaker-specific text file
                txt_path = output_dir / f"{base_name}_{speaker}.txt"
                text_content = "\n".join(segment["text"].strip() for segment in segments)
                txt_path.write_text(text_content, encoding='utf-8')

        # Save combined plain text (chronological order)
        all_segments = sorted(
            [s for segments in speaker_segments.values() for s in segments],
            key=lambda x: x["start"]
        )
        txt_path = output_dir / f"{base_name}.txt"
        txt_path.write_text("\n".join(s["text"].strip() for s in all_segments), encoding='utf-8')

        # Save detailed JSON
        json_path = output_dir / f"{base_name}.json"
        result.update({
            "speaker_segments": {
                speaker: [{
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"].strip(),
                    "speaker": speaker
                } for s in segments]
                for speaker, segments in speaker_segments.items()
            }
        })

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Log speaker statistics
        for speaker, segments in speaker_segments.items():
            if segments:
                total_duration = sum(s["end"] - s["start"] for s in segments)
                avg_length = total_duration / len(segments)
                self.logger.info(
                    f"{speaker}: {len(segments)} segments, "
                    f"{total_duration:.1f}s total duration, "
                    f"{avg_length:.1f}s avg length"
                )

    def cleanup(self):
        """Cleanup resources and finish monitoring"""
        self.logger.info("Cleaning up resources...")

        # Stop monitoring
        self.hardware_manager.stop_monitoring()
        self.process_monitor.stop_monitoring()

        # Cleanup transcription engine
        self.transcription_engine.cleanup()

        self.logger.info("Cleanup completed")

def get_audio_files() -> List[Path]:
    """Get list of audio files from audio_files directory"""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
    audio_dir = Path('audio_files')

    if not audio_dir.exists():
        print(f"\nCreating {audio_dir} directory...")
        audio_dir.mkdir(parents=True, exist_ok=True)
        return []

    audio_files = [
        f for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    return sorted(audio_files)

def select_files() -> List[Path]:
    """Interactive file selection"""
    audio_files = get_audio_files()

    if not audio_files:
        print("\nNo audio files found in audio_files directory.")
        while True:
            file_path = input("\nEnter the full path to an audio file (or 'q' to quit): ").strip()
            if file_path.lower() == 'q':
                sys.exit(0)

            path = Path(file_path)
            if path.exists() and path.is_file():
                return [path]
            print("File not found. Please try again.")

    print("\nAvailable audio files:")
    for i, file in enumerate(audio_files, 1):
        print(f"{i}. {file.name}")

    while True:
        choice = input("\nEnter file number(s) to process (comma-separated), 'all', or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            sys.exit(0)

        if choice.lower() == 'all':
            return audio_files

        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_files = [audio_files[i] for i in indices if 0 <= i < len(audio_files)]
            if selected_files:
                return selected_files
        except (ValueError, IndexError):
            pass

        print("Invalid selection. Please try again.")

def parse_arguments() -> Tuple[argparse.Namespace, ApplicationConfig]:
    """Parse command line arguments and get interactive input"""
    parser = argparse.ArgumentParser(
        description="Smart Audio Transcription System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add the server argument
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start in API server mode"
    )

    args = parser.parse_args()
    config = ApplicationConfig()

    # Get maximum recommended workers
    max_workers = get_max_workers()

    # Get GPU capabilities
    gpu_info = ConfigurationManager().get_gpu_capabilities()

    # Interactive selections
    print("\n=== Configuration Options ===")

    # Model selection
    config.model_size = select_model()

    # Language selection
    config.language = select_language()

    # Worker selection
    config.num_workers = select_workers(max_workers)

    # Device selection
    config.force_device = select_device(
        has_cuda=gpu_info["cuda"],
        has_mps=gpu_info.get("mps", False)
    )

    # Reports selection
    config.generate_reports = select_reports()

    # Verbose selection
    config.verbose = select_verbose()

    return args, config

def select_model() -> str:
    """Interactive model selection"""
    models = {
        "1": ("turbo", "Turbo - Fast and efficient (Default) [6GB VRAM]"),
        "2": ("base", "Base - Balanced performance [1GB VRAM]"),
        "3": ("small", "Small - Lower resource usage [2GB VRAM]"),
        "4": ("medium", "Medium - Better accuracy [5GB VRAM]"),
        "5": ("large", "Large - Best accuracy, high resource usage [10GB VRAM Required]")
    }

    print("\nAvailable models:")
    for key, (model, desc) in models.items():
        print(f"{key}. {desc}")

    while True:
        choice = input("\nSelect model number [1]: ").strip()
        if choice == "":
            return "turbo"
        if choice in models:
            model_name = models[choice][0]
            return model_name
        print("Invalid selection. Please try again.")

def select_language() -> str:
    """Interactive language selection"""
    languages = {
        "1": ("en", "English (Default)"),
        "2": ("fr", "French"),
        "3": ("es", "Spanish"),
        "4": ("ar", "Arabic")
    }

    print("\nAvailable languages:")
    for key, (code, name) in languages.items():
        print(f"{key}. {name}")

    while True:
        choice = input("\nSelect language number [1]: ").strip()
        if choice == "":
            return "en"  # Default to English
        if choice in languages:
            return languages[choice][0]
        print("Invalid selection. Please try again.")

def select_device(has_cuda: bool, has_mps: bool) -> str:
    """Interactive device selection"""
    devices = []
    if has_cuda:
        devices.append(("cuda", "NVIDIA GPU (Default)"))
    if has_mps:
        devices.append(("mps", "Apple Silicon"))
    devices.append(("cpu", "CPU" if not (has_cuda or has_mps) else "CPU (Default)"))

    print("\nAvailable devices:")
    for i, (device, desc) in enumerate(devices, 1):
        print(f"{i}. {desc}")

    while True:
        choice = input("\nSelect device number [1]: ").strip()
        if choice == "":
            return devices[0][0]  # Return first device as default
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(devices):
                return devices[idx][0]
        except ValueError:
            pass
        print("Invalid selection. Please try again.")

def main():
    """Main application entry point"""
    try:
        # Parse arguments first
        args, config = parse_arguments()

        # Check if server mode is requested
        if hasattr(args, 'server') and args.server:
            print("\nStarting API server...")
            start_api_server()
            return 0

        # Initialize system first (without input file)
        if not initialize_system():
            print("\nStartup cancelled. Exiting...")
            return 1

        # After system initialization, select files to process
        selected_files = select_files()
        if not selected_files:
            print("\nNo files selected. Exiting...")
            return 0

        # Update config with selected files
        config.input_paths = selected_files

        # Initialize transcriber with the first selected file
        transcriber = SmartTranscriber(audio_file=str(selected_files[0]))

        # Process files
        results = transcriber.process_files(config)

        # Print summary
        success_count = sum(1 for r in results.values() if "error" not in r)
        print(f"\nProcessing completed:")
        print(f"Successfully processed: {success_count}/{len(results)} files")

        return 0

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
