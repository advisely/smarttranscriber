# smarttranscriber/modules/hardwareManager.py

import psutil
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import platform
from enum import Enum, auto
import torch
from collections import deque
import logging
import io

# Import from Part 1
from modules.systemConfig import SystemType, ConfigurationManager

class HardwareType(Enum):
    CPU = auto()
    NVIDIA_GPU = auto()
    AMD_GPU = auto()
    INTEL_GPU = auto()
    APPLE_SILICON = auto()
    MEMORY = auto()
    STORAGE = auto()

@dataclass
class HardwareStats:
    usage_percent: float = 0.0
    temperature: float = 0.0
    memory_used: int = 0
    memory_total: int = 0
    power_draw: float = 0.0
    frequency: float = 0.0

@dataclass
class ResourceThresholds:
    cpu_temp_max: float = 95.0
    gpu_temp_max: float = 85.0
    memory_percent_max: float = 90.0
    gpu_memory_percent_max: float = 90.0
    power_draw_max: float = 300.0  # Watts

class HardwareManager:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = config_manager.logger
        self.system_type = config_manager.system_type

        # Initialize monitoring containers
        self.stats_history = {hw_type: deque(maxlen=60) for hw_type in HardwareType}
        self.current_stats = {hw_type: HardwareStats() for hw_type in HardwareType}
        self.thresholds = ResourceThresholds()

        # Initialize hardware monitors
        self._init_hardware_monitors()

        # Start monitoring
        self.monitoring = False
        self.monitoring_thread = None

        self.logger.info("Hardware Manager initialized")
        self.config_manager.add_to_report("Hardware Manager initialized", "INFO")

    def _monitor_cpu_basic(self) -> HardwareStats:
        """Basic CPU monitoring that works across all platforms"""
        try:
            # Get CPU usage percentage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Get CPU frequency
            cpu_freq = psutil.cpu_freq()
            frequency = cpu_freq.current if cpu_freq else 0.0

            # Get memory information
            memory = psutil.virtual_memory()

            # Try to get CPU temperature, default to 0 if not available
            try:
                temperatures = psutil.sensors_temperatures()
                # Look for CPU temperature from different possible keys
                temp_keys = ['coretemp', 'cpu_thermal', 'cpu-thermal', 'k10temp']
                temp = 0.0
                for key in temp_keys:
                    if key in temperatures and temperatures[key]:
                        temp = temperatures[key][0].current
                        break
            except (AttributeError, KeyError):
                temp = 0.0

            return HardwareStats(
                usage_percent=cpu_percent,
                temperature=temp,
                memory_used=memory.used,
                memory_total=memory.total,
                power_draw=0.0,  # Basic monitoring doesn't provide power info
                frequency=frequency
            )

        except Exception as e:
            self.logger.error(f"Error monitoring CPU: {e}")
            return HardwareStats()

    def _monitor_cpu_windows(self) -> HardwareStats:
        """Windows-specific CPU monitoring"""
        try:
            # Get basic stats first
            stats = self._monitor_cpu_basic()

            # Try to get temperature through WMI
            try:
                if hasattr(self, 'wmi'):
                    temps = self.wmi.MSAcpi_ThermalZoneTemperature()
                    if temps:
                        # Convert temperature from deciKelvin to Celsius
                        stats.temperature = (temps[0].CurrentTemperature / 10.0) - 273.15
            except Exception as e:
                self.logger.debug(f"Error getting Windows CPU temperature: {e}")

            return stats

        except Exception as e:
            self.logger.error(f"Error monitoring Windows CPU: {e}")
            return self._monitor_cpu_basic()

    def _monitor_cpu_linux(self) -> HardwareStats:
        """Linux-specific CPU monitoring"""
        return self._monitor_cpu_basic()  # Linux basic monitoring is already comprehensive

    def _monitor_cpu_macos(self) -> HardwareStats:
        """macOS-specific CPU monitoring"""
        return self._monitor_cpu_basic()  # macOS basic monitoring is sufficient

    def _monitor_memory(self) -> HardwareStats:
        """Monitor system memory"""
        try:
            memory = psutil.virtual_memory()
            return HardwareStats(
                usage_percent=memory.percent,
                memory_used=memory.used,
                memory_total=memory.total
            )
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {e}")
            return HardwareStats()

    def _monitor_nvidia_gpu(self) -> HardwareStats:
        """Monitor NVIDIA GPU using NVML"""
        try:
            if not hasattr(self, 'nvml') or not hasattr(self, 'nvidia_handle'):
                return HardwareStats()

            # Get memory information
            info = self.nvml.nvmlDeviceGetMemoryInfo(self.nvidia_handle)

            # Get temperature
            temp = self.nvml.nvmlDeviceGetTemperature(self.nvidia_handle, self.nvml.NVML_TEMPERATURE_GPU)

            # Get utilization
            utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.nvidia_handle)
            gpu_util = utilization.gpu if hasattr(utilization, 'gpu') else 0

            # Get power usage (optional)
            try:
                power = self.nvml.nvmlDeviceGetPowerUsage(self.nvidia_handle) / 1000.0
            except:
                power = 0.0

            return HardwareStats(
                usage_percent=float(gpu_util),
                temperature=float(temp),
                memory_used=info.used,
                memory_total=info.total,
                power_draw=power
            )
        except Exception as e:
            self.logger.debug(f"Error monitoring NVIDIA GPU: {e}")
            return HardwareStats()

    def _init_hardware_monitors(self):
        """Initialize hardware-specific monitoring capabilities"""
        self.monitors = {}

        # CPU Monitor
        self.monitors[HardwareType.CPU] = self._init_cpu_monitor()

        # GPU Monitors
        if self.system_type == SystemType.WINDOWS:
            self._init_windows_gpu_monitors()
        elif self.system_type == SystemType.LINUX:
            self._init_linux_gpu_monitors()
        elif self.system_type == SystemType.MACOS:
            self._init_macos_gpu_monitors()

        # Memory Monitor
        self.monitors[HardwareType.MEMORY] = self._monitor_memory

        self.logger.info("Hardware monitors initialized")

    def _init_cpu_monitor(self):
        """Initialize CPU monitoring based on platform"""
        if self.system_type == SystemType.WINDOWS:
            try:
                import wmi
                self.wmi = wmi.WMI()
                return self._monitor_cpu_windows
            except ImportError:
                self.logger.warning("WMI not available, using basic CPU monitoring")
                return self._monitor_cpu_basic
        elif self.system_type == SystemType.LINUX:
            return self._monitor_cpu_linux
        elif self.system_type == SystemType.MACOS:
            return self._monitor_cpu_macos
        else:
            return self._monitor_cpu_basic

    def _init_windows_gpu_monitors(self):
        """Initialize GPU monitoring for Windows"""
        # NVIDIA GPU
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            self.nvml = nvml
            self.monitors[HardwareType.NVIDIA_GPU] = self._monitor_nvidia_gpu
            self.logger.info("NVIDIA GPU monitoring initialized")
        except Exception as e:
            self.logger.debug(f"NVIDIA GPU monitoring not available: {e}")

    def _init_linux_gpu_monitors(self):
        """Initialize GPU monitoring for Linux/WSL"""
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)

            # Test GPU monitoring functions
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)

            self.nvml = nvml
            self.nvidia_handle = handle
            self.monitors[HardwareType.NVIDIA_GPU] = self._monitor_nvidia_gpu
            self.logger.info(f"NVIDIA GPU monitoring initialized: {nvml.nvmlDeviceGetName(handle).decode()}")
        except Exception as e:
            self.logger.debug(f"NVIDIA GPU monitoring not available: {e}")

    def _init_macos_gpu_monitors(self):
        """Initialize GPU monitoring for macOS"""
        if platform.processor() == 'arm':
            self.monitors[HardwareType.APPLE_SILICON] = self._monitor_basic_gpu
            self.logger.info("Apple Silicon monitoring initialized")

    def _monitor_basic_gpu(self) -> HardwareStats:
        """Basic GPU monitoring when specific APIs aren't available"""
        return HardwareStats()

    def start_monitoring(self):
        """Start hardware monitoring in a separate thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Hardware monitoring started")

    def stop_monitoring(self):
        """Stop hardware monitoring"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            self.logger.info("Hardware monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._update_all_stats()
                self._check_thresholds()
                time.sleep(1)  # Update frequency
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)  # Longer delay on error

    def _update_all_stats(self):
        """Update statistics for all monitored hardware"""
        for hw_type, monitor in self.monitors.items():
            try:
                stats = monitor()
                self.current_stats[hw_type] = stats
                self.stats_history[hw_type].append(stats)
            except Exception as e:
                self.logger.debug(f"Error monitoring {hw_type}: {e}")

    def _check_thresholds(self):
        """Check if any hardware metrics exceed thresholds"""
        # CPU temperature check
        if self.current_stats[HardwareType.CPU].temperature > self.thresholds.cpu_temp_max:
            self.logger.warning("CPU temperature exceeded threshold!")
            self.config_manager.add_to_report(
                f"CPU temperature warning: {self.current_stats[HardwareType.CPU].temperature}°C",
                "WARNING"
            )

        # GPU checks
        for gpu_type in [HardwareType.NVIDIA_GPU, HardwareType.AMD_GPU,
                        HardwareType.INTEL_GPU, HardwareType.APPLE_SILICON]:
            if gpu_type in self.current_stats:
                stats = self.current_stats[gpu_type]
                if stats.temperature > self.thresholds.gpu_temp_max:
                    self.logger.warning(f"{gpu_type} temperature exceeded threshold!")
                if stats.memory_total > 0 and (stats.memory_used / stats.memory_total * 100) > self.thresholds.gpu_memory_percent_max:
                    self.logger.warning(f"{gpu_type} memory usage exceeded threshold!")

        # Memory check
        mem = psutil.virtual_memory()
        if mem.percent > self.thresholds.memory_percent_max:
            self.logger.warning("System memory usage exceeded threshold!")

    def get_optimal_device(self) -> Tuple[str, Dict[str, Any]]:
        """
        Determine the optimal processing device based on hardware availability and status
        Returns: (device_type, device_info)
        """
        available_devices = []

        # First check CUDA availability explicitly
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality
                torch.cuda.current_device()
                device_info = {
                    'type': 'cuda',
                    'name': torch.cuda.get_device_name(0),
                    'memory': torch.cuda.get_device_properties(0).total_memory,
                    'temperature': 0,  # Could be updated if monitoring is working
                    'usage': 0
                }
                return 'cuda', device_info
            except Exception as e:
                self.logger.warning(f"CUDA device detected but not accessible: {e}")

        # Check NVIDIA GPU
        if HardwareType.NVIDIA_GPU in self.current_stats:
            stats = self.current_stats[HardwareType.NVIDIA_GPU]
            if stats.memory_total > 0:  # GPU is available
                available_devices.append({
                    'type': 'cuda',
                    'name': 'NVIDIA GPU',
                    'memory': stats.memory_total,
                    'temperature': stats.temperature,
                    'usage': stats.usage_percent
                })

        # Exclude 'mps' unless on Apple Silicon
        if self.system_type == SystemType.MACOS and platform.processor() == 'arm':
            # Check Apple Silicon
            if HardwareType.APPLE_SILICON in self.current_stats:
                stats = self.current_stats[HardwareType.APPLE_SILICON]
                available_devices.append({
                    'type': 'mps',
                    'name': 'Apple Silicon',
                    'memory': stats.memory_total,
                    'temperature': stats.temperature,
                    'usage': stats.usage_percent
                })

        # Select best device
        if available_devices:
            # Sort by available memory and temperature
            available_devices.sort(
                key=lambda x: (x['memory'], -x['temperature'], -x['usage']),
                reverse=True
            )
            best_device = available_devices[0]
            return best_device['type'], best_device

        # Fallback to CPU
        cpu_stats = self.current_stats[HardwareType.CPU]
        return 'cpu', {
            'type': 'cpu',
            'name': platform.processor(),
            'memory': psutil.virtual_memory().total,
            'temperature': cpu_stats.temperature,
            'usage': cpu_stats.usage_percent
        }

    def get_recommended_batch_size(self, device_type: str) -> int:
        """Determine recommended batch size based on available memory"""
        if device_type == 'cpu':
            available_ram = psutil.virtual_memory().available
            return max(1, min(8, available_ram // (1024**3)))  # 1GB per batch
        else:
            for gpu_type in [HardwareType.NVIDIA_GPU, HardwareType.AMD_GPU,
                           HardwareType.APPLE_SILICON]:
                if gpu_type in self.current_stats:
                    stats = self.current_stats[gpu_type]
                    available_memory = stats.memory_total - stats.memory_used
                    return max(1, min(16, available_memory // (1024**3)))  # 1GB per batch
        return 1  # Default safe value

    def generate_hardware_report(self) -> Dict[str, Any]:
        """Generate comprehensive hardware status report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_type': self.system_type.name,
            'devices': {},
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent_used': psutil.virtual_memory().percent
            },
            'monitoring_status': self.monitoring,
            'recommendations': {}
        }

        # Add device-specific information
        for hw_type, stats in self.current_stats.items():
            if stats.memory_total > 0:  # Device is available
                report['devices'][hw_type.name] = {
                    'temperature': stats.temperature,
                    'usage_percent': stats.usage_percent,
                    'memory_used': stats.memory_used,
                    'memory_total': stats.memory_total,
                    'power_draw': stats.power_draw
                }

        # Add recommendations
        optimal_device, device_info = self.get_optimal_device()
        report['recommendations'] = {
            'optimal_device': optimal_device,
            'device_info': device_info,
            'batch_size': self.get_recommended_batch_size(optimal_device),
            'warnings': [msg for msg in self.logger.handlers[0].stream.getvalue().split('\n')
                        if 'WARNING' in msg]
        }

        return report

