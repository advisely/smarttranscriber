"""
Smart Transcriber System Handler
Handles system detection, configuration, and initialization
Part 1 of 5
"""

import platform
import sys
import os
import logging
import json
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import shutil
from datetime import datetime
import subprocess
from functools import lru_cache
import psutil

class SystemType(Enum):
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    WSL = auto()
    UNKNOWN = auto()

@dataclass
class SystemRequirements:
    min_python_version: tuple = (3, 7)
    min_ram_gb: int = 4
    min_cpu_cores: int = 2
    recommended_ram_gb: int = 8
    recommended_cpu_cores: int = 4

@dataclass
class SystemCapabilities:
    cuda_available: bool = False
    rocm_available: bool = False
    mps_available: bool = False
    directml_available: bool = False
    ipex_available: bool = False
    numpy_available: bool = False
    torch_available: bool = False
    ffmpeg_available: bool = False

@dataclass
class SystemStatus:
    timestamp: str
    system_type: SystemType
    python_version: str
    cpu_info: Dict[str, Any]
    ram_info: Dict[str, Any]
    gpu_info: List[Dict[str, Any]]
    capabilities: SystemCapabilities
    requirements_met: bool
    initialization_successful: bool
    warnings: List[str]
    errors: List[str]

@dataclass        # <-- Add the new class here
class SystemInfo:
    system_type: SystemType
    python_version: str
    cpu_info: Dict[str, Any]
    ram_info: Dict[str, Any]

class ConfigurationManager:
    def __init__(self, audio_file=None):
            # Add audio_file parameter and create timestamp
            self.audio_file = audio_file
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Create output directory structure
            if audio_file:
                self.output_dir = Path("output") / f"{Path(audio_file).stem}_{self.timestamp}"
                self.output_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.output_dir = Path("output")
                self.output_dir.mkdir(exist_ok=True)

            # Initialize report_content first
            self.report_content = []

            # Set up logger
            self.logger = self._setup_logger()
            self.start_time = datetime.now()

            # Initialize requirements and capabilities before system type detection
            self.requirements = SystemRequirements()
            self.capabilities = SystemCapabilities()

            # Now detect system type and initialize status
            self.system_type = self._detect_system_type()
            self.status = self._initialize_status()

            # Add initial report entry
            self.add_to_report("System Configuration Started", level="INFO")

            self.system_info = SystemInfo(
                        system_type=self.system_type,
                        python_version=platform.python_version(),
                        cpu_info=self._get_cpu_info(),
                        ram_info=self._get_ram_info()
            )

    def _initialize_status(self) -> SystemStatus:
        """Initialize system status with current system information"""
        try:
            # Get CPU information
            cpu_info = {
                "processor": platform.processor(),
                "cores": os.cpu_count(),
                "architecture": platform.machine()
            }

            # Get RAM information
            memory = psutil.virtual_memory()
            ram_info = {
                "total": memory.total,
                "available": memory.available,
                "percent_used": memory.percent
            }

            # Initialize status
            status = SystemStatus(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                system_type=self.system_type,
                python_version=platform.python_version(),
                cpu_info=cpu_info,
                ram_info=ram_info,
                gpu_info=[],  # Will be populated later by get_gpu_capabilities
                capabilities=self.capabilities,
                requirements_met=True,  # Will be updated after checks
                initialization_successful=True,
                warnings=[],
                errors=[]
            )

            self.add_to_report("System status initialized", level="INFO")
            return status

        except Exception as e:
            self.add_to_report(f"Error initializing system status: {e}", level="ERROR")
            # Return a minimal status object in case of error
            return SystemStatus(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                system_type=self.system_type,
                python_version=platform.python_version(),
                cpu_info={},
                ram_info={},
                gpu_info=[],
                capabilities=self.capabilities,
                requirements_met=False,
                initialization_successful=False,
                warnings=[],
                errors=[str(e)]
            )

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('SmartTranscriber')
        logger.setLevel(logging.INFO)

        # File handler - now in output directory
        log_file = self.output_dir / 'transcriber_system.log'
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(ch)

        return logger

    def _detect_system_type(self) -> SystemType:
        """Detect the operating system and environment"""
        self.add_to_report("Detecting System Type", level="INFO")

        system = platform.system().lower()
        if system == 'linux':
            # Check for WSL
            try:
                with open('/proc/version', 'r') as f:
                    if 'microsoft' in f.read().lower():
                        self.add_to_report("Detected WSL Environment", level="INFO")
                        return SystemType.WSL
            except Exception as e:
                self.add_to_report(f"Error checking WSL: {e}", level="DEBUG")

            self.add_to_report("Detected Linux System", level="INFO")
            return SystemType.LINUX

        elif system == 'windows':
            self.add_to_report("Detected Windows System", level="INFO")
            return SystemType.WINDOWS

        elif system == 'darwin':
            self.add_to_report("Detected macOS System", level="INFO")
            return SystemType.MACOS

        self.add_to_report("Unable to determine system type", level="WARNING")
        return SystemType.UNKNOWN

    def add_to_report(self, message: str, level: str = "INFO"):
        """Add a message to the configuration report"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.report_content.append({
            "timestamp": timestamp,
            "level": level,
            "message": message
        })

        # Also log to system logger
        log_level = getattr(logging, level)
        self.logger.log(log_level, message)

    @lru_cache(maxsize=None)
    def check_command(self, command: str) -> bool:
        """Check if a command is available in the system"""
        return shutil.which(command) is not None

    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is installed and accessible"""
        self.add_to_report("Checking FFmpeg installation", level="INFO")

        ffmpeg_available = self.check_command('ffmpeg')
        if ffmpeg_available:
            try:
                result = subprocess.run(
                    ['ffmpeg', '-version'],
                    capture_output=True,
                    text=True
                )
                version = result.stdout.split('\n')[0]
                self.add_to_report(f"FFmpeg found: {version}", level="INFO")
                return True
            except Exception as e:
                self.add_to_report(f"FFmpeg error: {e}", level="WARNING")
                return False
        else:
            self.add_to_report("FFmpeg not found", level="WARNING")
            return False

    def check_python_compatibility(self) -> bool:
        """Check Python version compatibility"""
        current_version = sys.version_info[:2]
        min_version = self.requirements.min_python_version

        self.add_to_report(
            f"Checking Python version (minimum: {min_version[0]}.{min_version[1]})",
            level="INFO"
        )

        if current_version >= min_version:
            self.add_to_report(
                f"Python version {current_version[0]}.{current_version[1]} is compatible",
                level="INFO"
            )
            return True
        else:
            self.add_to_report(
                f"Python version {current_version[0]}.{current_version[1]} is below minimum required",
                level="ERROR"
            )
            return False

    def get_gpu_capabilities(self) -> Dict[str, Any]:
        """Detect and verify GPU capabilities"""
        self.add_to_report("Checking GPU capabilities", level="INFO")

        capabilities = {
            "cuda": False,
            "rocm": False,
            "mps": False,
            "directml": False,
            "ipex": False,
            "detected_gpus": []
        }

        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                capabilities["cuda"] = True
                for i in range(torch.cuda.device_count()):
                    gpu_info = {
                        "name": torch.cuda.get_device_name(i),
                        "memory": torch.cuda.get_device_properties(i).total_memory,
                        "capability": f"{torch.cuda.get_device_capability(i)}"
                    }
                    capabilities["detected_gpus"].append(gpu_info)
                    self.add_to_report(
                        f"Found CUDA GPU: {gpu_info['name']}",
                        level="INFO"
                    )
        except Exception as e:
            self.add_to_report(f"CUDA check error: {e}", level="DEBUG")

        # Check ROCm
        try:
            if hasattr(torch, 'has_rocm') and torch.has_rocm:
                capabilities["rocm"] = True
                self.add_to_report("ROCm support detected", level="INFO")
        except Exception as e:
            self.add_to_report(f"ROCm check error: {e}", level="DEBUG")

        # Check MPS (Apple Silicon)
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                capabilities["mps"] = True
                self.add_to_report("Apple Silicon MPS support detected", level="INFO")
        except Exception as e:
            self.add_to_report(f"MPS check error: {e}", level="DEBUG")

        return capabilities

    def initialize_system_config(self) -> Dict[str, Any]:
        """Initialize complete system configuration"""
        config = {
            "system_type": self.system_type,
            "python_compatible": self.check_python_compatibility(),
            "ffmpeg_available": self.check_ffmpeg(),
            "gpu_capabilities": self.get_gpu_capabilities(),
            "environment_variables": self._setup_environment_variables(),
            "temp_directory": self._setup_temp_directory(),
            "initialization_time": str(datetime.now() - self.start_time)
        }

        self.add_to_report("System configuration completed", level="INFO")
        return config

    def _setup_environment_variables(self) -> Dict[str, str]:
        """Setup necessary environment variables based on system type"""
        env_vars = {}

        if self.system_type == SystemType.WINDOWS:
            env_vars.update(self._setup_windows_env_vars())
        elif self.system_type in [SystemType.LINUX, SystemType.WSL]:
            env_vars.update(self._setup_linux_env_vars())
        elif self.system_type == SystemType.MACOS:
            env_vars.update(self._setup_macos_env_vars())

        return env_vars

    def _setup_windows_env_vars(self) -> Dict[str, str]:
        """Setup Windows-specific environment variables"""
        env_vars = {}

        # CUDA paths
        if 'CUDA_PATH' in os.environ:
            cuda_path = os.environ['CUDA_PATH']
            env_vars['PATH'] = f"{cuda_path}\\bin;{os.environ.get('PATH', '')}"

        # Intel OneAPI
        oneapi_path = "C:\\Program Files (x86)\\Intel\\oneAPI"
        if os.path.exists(oneapi_path):
            env_vars['ONEAPI_ROOT'] = oneapi_path

        self.add_to_report("Windows environment variables configured", level="INFO")
        return env_vars

    def _setup_linux_env_vars(self) -> Dict[str, str]:
        """Setup Linux-specific environment variables"""
        env_vars = {}

        # CUDA paths
        cuda_paths = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/bin"
        ]

        # ROCm paths
        rocm_paths = [
            "/opt/rocm/lib",
            "/opt/rocm/bin"
        ]

        # Combine paths
        existing_paths = [p for p in cuda_paths + rocm_paths if os.path.exists(p)]
        if existing_paths:
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            env_vars['LD_LIBRARY_PATH'] = ':'.join(
                [current_ld_path] + existing_paths
            )

        self.add_to_report("Linux environment variables configured", level="INFO")
        return env_vars

    def _setup_macos_env_vars(self) -> Dict[str, str]:
        """Setup macOS-specific environment variables"""
        env_vars = {}

        # Check for Apple Silicon
        if platform.processor() == 'arm':
            env_vars['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        self.add_to_report("macOS environment variables configured", level="INFO")
        return env_vars

    def _setup_temp_directory(self) -> str:
        """Setup and verify temporary directory"""
        if self.system_type == SystemType.WINDOWS:
            temp_dir = Path(os.environ.get('TEMP', 'C:\\Windows\\Temp'))
        else:
            temp_dir = Path('/tmp')

        # Create transcriber-specific temp directory
        temp_dir = temp_dir / 'smart_transcriber'
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.add_to_report(f"Temporary directory setup at: {temp_dir}", level="INFO")
        return str(temp_dir)

    def generate_report(self) -> str:
        """Generate a formatted report of the configuration process"""
        report_lines = [
            "=== Smart Transcriber Configuration Report ===",
            f"Configuration Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"System Type: {self.system_type.name}",
            "\nConfiguration Steps:",
        ]

        for entry in self.report_content:
            report_lines.append(
                f"[{entry['timestamp']}] {entry['level']}: {entry['message']}"
            )

        return "\n".join(report_lines)

    def save_report(self, filename: str = "transcriber_config_report.txt"):
        """Save the configuration report to a file"""
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        self.logger.info(f"Configuration report saved to {filename}")

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        return {
            "processor": platform.processor(),
            "cores": os.cpu_count(),
            "architecture": platform.machine()
        }

    def _get_ram_info(self) -> Dict[str, Any]:
        """Get RAM information"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "percent_used": memory.percent
        }

