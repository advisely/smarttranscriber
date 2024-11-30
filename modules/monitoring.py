# smarttranscriber/modules/monitoring.py

"""
Smart Transcriber Monitoring & Reporting
Handles real-time monitoring, analysis, and report generation
Part 4 of 5
"""

import torch
import curses
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
import threading
from queue import Queue
import humanize
import logging
import pandas as pd
import matplotlib.pyplot as plt
import psutil

# Import from previous parts
from modules.systemConfig import ConfigurationManager, SystemType
from modules.hardwareManager import HardwareManager, HardwareType, HardwareStats
from modules.transcriptionEngine import TranscriptionStats, TranscriptionEngine

@dataclass
class ProcessingEvent:
    timestamp: float
    event_type: str
    message: str
    level: str
    details: Optional[Dict] = None

@dataclass
class ProgressStats:
    total_files: int = 0
    processed_files: int = 0
    current_file: str = ""
    current_progress: float = 0.0
    estimated_time_remaining: float = 0.0
    average_speed: float = 0.0
    total_audio_duration: float = 0.0
    total_processed_duration: float = 0.0

class MonitoringDisplay:
    def __init__(self):
        # Initialize last_progress
        self.last_progress = ProgressStats(
            total_files=0,
            processed_files=0,
            current_file="",
            current_progress=0.0
        )

        try:
            # Try to initialize curses
            self.use_curses = True
            self.stdscr = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
            curses.noecho()
            curses.cbreak()
            self.stdscr.nodelay(1)
            self.stdscr.keypad(1)

            # Initialize display areas
            self.max_y, self.max_x = self.stdscr.getmaxyx()
            self.progress_window = curses.newwin(5, self.max_x, 0, 0)
            self.hardware_window = curses.newwin(10, self.max_x, 5, 0)
            self.stats_window = curses.newwin(5, self.max_x, 15, 0)
            self.log_window = curses.newwin(self.max_y - 20, self.max_x, 20, 0)
        except:
            # Fallback to simple print-based display
            self.use_curses = False
            self.cleanup()

    def cleanup(self, stats: ProgressStats):
        """Restore terminal settings and show final summary"""
        if self.use_curses:
            try:
                curses.nocbreak()
                self.stdscr.keypad(0)
                curses.echo()
                curses.endwin()
            except:
                pass

        # Clear the screen and move cursor to top before final output
        print('\033[2J\033[H', end='')
        print("\nProcessing Summary:")
        print("-" * 50)
        print(f"Status: Completed")
        print(f"Files Processed: {stats.processed_files}/{stats.total_files}")
        print("-" * 50)

    def update_progress(self, stats: ProgressStats):
        """Update progress display"""
        self.last_progress = stats  # Store for final display
        if not self.use_curses:
            # Clear previous lines and show clean progress
            print('\033[2J\033[H', end='')  # Clear screen and move to top
            print("=== Processing Status ===")
            print(f"File: {stats.current_file}")

            # Progress bar
            bar_width = 50
            filled = int(bar_width * stats.current_progress)
            bar = f"[{'=' * filled}{' ' * (bar_width - filled)}]"
            print(f"Progress: {bar} {stats.current_progress*100:.1f}%")
            print(f"Files: {stats.processed_files}/{stats.total_files}")
            if stats.estimated_time_remaining > 0:
                print(f"ETA: {timedelta(seconds=int(stats.estimated_time_remaining))}")
            print()  # Empty line before hardware stats
            return

        self.progress_window.clear()
        self.progress_window.addstr(0, 0, "=== Transcription Progress ===",
                                  curses.A_BOLD)

        # Progress bar
        progress = stats.current_progress
        bar_width = self.max_x - 20
        filled = int(bar_width * progress)
        bar = f"[{'=' * filled}{' ' * (bar_width - filled)}]"

        self.progress_window.addstr(1, 0,
            f"Processing: {stats.current_file}")
        self.progress_window.addstr(2, 0,
            f"Progress: {bar} {progress*100:.1f}%")
        self.progress_window.addstr(3, 0,
            f"Files: {stats.processed_files}/{stats.total_files} | "
            f"ETA: {timedelta(seconds=int(stats.estimated_time_remaining))}")

        self.progress_window.refresh()

    def update_hardware(self, hw_stats: Dict[HardwareType, HardwareStats]):
        """Update hardware statistics display"""
        if not self.use_curses:
            # Clear previous lines
            print('\033[2J\033[H', end='')  # Clear screen and move to top
            print("=== Hardware Status ===")
            for hw_type, stats in hw_stats.items():
                print(f"{hw_type.name:<15}: "
                    f"Usage: {stats.usage_percent:5.1f}% | "
                    f"Temp: {stats.temperature:5.1f}°C | "
                    f"Memory: {stats.memory_used/1024**3:5.1f}/"
                    f"{stats.memory_total/1024**3:5.1f}GB")
            return

        # Original curses code remains unchanged
        self.hardware_window.clear()
        self.hardware_window.addstr(0, 0, "=== Hardware Status ===",
                                curses.A_BOLD)

        y = 1
        for hw_type, stats in hw_stats.items():
            color = (curses.color_pair(1) if stats.usage_percent < 70 else
                    curses.color_pair(2) if stats.usage_percent < 90 else
                    curses.color_pair(3))

            self.hardware_window.addstr(y, 0, f"{hw_type.name}:")
            self.hardware_window.addstr(y, 15,
                f"Usage: {stats.usage_percent:5.1f}% | "
                f"Temp: {stats.temperature:5.1f}°C | "
                f"Memory: {stats.memory_used/1024**3:5.1f}/"
                f"{stats.memory_total/1024**3:5.1f}GB", color)
            y += 1

        self.hardware_window.refresh()

    def update_stats(self, stats: TranscriptionStats):
        """Update transcription statistics display"""
        if not self.use_curses:
            if stats.end_time > 0:
                speed = stats.audio_duration / stats.processing_duration
                print(f"\rSpeed: {speed:.2f}x | "
                    f"Peak Memory: {humanize.naturalsize(stats.peak_memory)} | "
                    f"GPU Memory: {humanize.naturalsize(stats.peak_gpu_memory)} | "
                    f"Model: {stats.model_size} | Device: {stats.device_used}")
            return

        # Original curses code remains unchanged
        self.stats_window.clear()
        self.stats_window.addstr(0, 0, "=== Processing Statistics ===",
                            curses.A_BOLD)

        if stats.end_time > 0:
            speed = stats.audio_duration / stats.processing_duration
            self.stats_window.addstr(1, 0,
                f"Speed: {speed:.2f}x | "
                f"Peak Memory: {humanize.naturalsize(stats.peak_memory)} | "
                f"GPU Memory: {humanize.naturalsize(stats.peak_gpu_memory)}")
            self.stats_window.addstr(2, 0,
                f"Model: {stats.model_size} | Device: {stats.device_used}")

        self.stats_window.refresh()

    def update_log(self, events: List[ProcessingEvent]):
        """Update log display with recent events"""
        if not self.use_curses:
            # In non-curses mode, just print the most recent event
            if events:
                latest = events[-1]
                print(f"\n[{datetime.fromtimestamp(latest.timestamp).strftime('%H:%M:%S')}] "
                    f"[{latest.level}] {latest.message}")
            return

        # Original curses code remains unchanged
        self.log_window.clear()
        self.log_window.addstr(0, 0, "=== Recent Events ===", curses.A_BOLD)

        max_events = self.max_y - 22
        recent_events = events[-max_events:] if len(events) > max_events else events

        for i, event in enumerate(recent_events, 1):
            color = (curses.color_pair(1) if event.level == "INFO" else
                    curses.color_pair(2) if event.level == "WARNING" else
                    curses.color_pair(3) if event.level == "ERROR" else
                    curses.color_pair(4))

            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            self.log_window.addstr(i, 0,
                f"{timestamp} [{event.level}] {event.message}", color)

        self.log_window.refresh()

    def update_progress(self, stats: ProgressStats):
        """Update progress display"""
        if not self.use_curses:
            # Create a simple progress bar
            bar_width = 50
            filled = int(bar_width * stats.current_progress)
            bar = f"[{'=' * filled}{' ' * (bar_width - filled)}]"
            print(f"\rProcessing: {stats.current_file}\n"
                f"Progress: {bar} {stats.current_progress*100:.1f}%\n"
                f"Files: {stats.processed_files}/{stats.total_files} | "
                f"ETA: {timedelta(seconds=int(stats.estimated_time_remaining))}", end='')
            return

        # Original curses code remains unchanged
        self.progress_window.clear()
        self.progress_window.addstr(0, 0, "=== Transcription Progress ===",
                                curses.A_BOLD)

        progress = stats.current_progress
        bar_width = self.max_x - 20
        filled = int(bar_width * progress)
        bar = f"[{'=' * filled}{' ' * (bar_width - filled)}]"

        self.progress_window.addstr(1, 0,
            f"Processing: {stats.current_file}")
        self.progress_window.addstr(2, 0,
            f"Progress: {bar} {progress*100:.1f}%")
        self.progress_window.addstr(3, 0,
            f"Files: {stats.processed_files}/{stats.total_files} | "
            f"ETA: {timedelta(seconds=int(stats.estimated_time_remaining))}")

        self.progress_window.refresh()

class PerformanceAnalyzer:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = config_manager.logger
        self.stats_history: List[Dict] = []
        self.event_history: List[ProcessingEvent] = []

    def add_stats(self, stats: Dict):
        """Add performance statistics to history"""
        stats['timestamp'] = time.time()
        self.stats_history.append(stats)

    def add_event(self, event: ProcessingEvent):
        """Add processing event to history"""
        self.event_history.append(event)

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance analysis"""
        if not self.stats_history:
            self.logger.warning("No performance data to generate report.")
            return {}

        df = pd.DataFrame(self.stats_history)

        # Avoid division by zero
        total_processing_time = df['processing_duration'].sum()
        if total_processing_time == 0:
            average_speed = 0
        else:
            average_speed = df['audio_duration'].sum() / total_processing_time

        analysis = {
            'processing_summary': {
                'total_duration': df['audio_duration'].sum(),
                'total_processing_time': total_processing_time,
                'average_speed': average_speed,
                'files_processed': len(df)
            },
            'resource_usage': {
                'peak_memory': df['peak_memory'].max(),
                'peak_gpu_memory': df['peak_gpu_memory'].max(),
                'average_cpu_usage': df['cpu_usage'].mean(),
                'average_gpu_usage': df['gpu_usage'].mean() if 'gpu_usage' in df else 0
            },
            'error_analysis': {
                'total_errors': len([e for e in self.event_history if e.level == "ERROR"]),
                'error_rate': len([e for e in self.event_history if e.level == "ERROR"]) / len(df) if len(df) > 0 else 0
            }
        }

        return analysis

    def generate_plots(self, output_dir: Path):
        """Generate performance visualization plots"""
        if not self.stats_history:
            return

        df = pd.DataFrame(self.stats_history)

        # Convert timestamp to relative time in seconds
        start_time = df['timestamp'].min()
        df['relative_time'] = df['timestamp'] - start_time

        # Processing Speed Plot
        plt.figure(figsize=(12, 6))
        plt.plot(df['relative_time'], df['audio_duration'] / df['processing_duration'])
        plt.title('Processing Speed Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (x)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'speed_plot.png')
        plt.close()

        # Resource Usage Plot
        plt.figure(figsize=(12, 6))
        plt.plot(df['relative_time'], df['peak_memory'] / (1024**3),
                label='Memory (GB)', color='blue')

        if 'peak_gpu_memory' in df.columns and df['peak_gpu_memory'].max() > 0:
            plt.plot(df['relative_time'], df['peak_gpu_memory'] / (1024**3),
                    label='GPU Memory (GB)', color='red')

        plt.title('Resource Usage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (GB)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'resource_plot.png')
        plt.close()

class ProcessMonitor:
    def __init__(self, config_manager: ConfigurationManager,
                 hardware_manager: HardwareManager,
                 performance_analyzer: 'PerformanceAnalyzer'):
        self.config_manager = config_manager
        self.hardware_manager = hardware_manager
        self.logger = config_manager.logger
        self.performance_analyzer = performance_analyzer

        self.display = MonitoringDisplay()
        self.analyzer = performance_analyzer

        # Initialize with None values until set_total_files is called
        self.progress_stats = ProgressStats(
            total_files=0,
            processed_files=0
        )
        self.events: List[ProcessingEvent] = []
        self.start_time = time.time()
        self.stop_event = threading.Event()

    def set_total_files(self, total: int):
        """Set the total number of files to be processed"""
        self.progress_stats.total_files = total
        self.progress_stats.processed_files = 0

    def start_monitoring(self):
        """Start the monitoring display"""
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        self.stop_event.set()
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join()
        # Pass the progress_stats to the cleanup method
        self.display.cleanup(self.progress_stats)

    def cleanup(self):
        """Clean up and display final summary"""
        self.display.cleanup(self.progress_stats)

    def _monitor_loop(self):
        """Main monitoring loop"""
        update_interval = 0.1  # seconds

        while not self.stop_event.is_set():
            try:
                # Update hardware stats
                hw_stats = self.hardware_manager.current_stats
                self.display.update_hardware(hw_stats)

                # Update progress
                self.display.update_progress(self.progress_stats)

                # Update events log
                self.display.update_log(self.events[-50:])  # Show last 50 events

                time.sleep(update_interval)

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(1)

    def update_progress(self, current_file: str, progress: float):
        """Update progress information"""
        self.progress_stats.current_file = current_file
        self.progress_stats.current_progress = progress

        if progress >= 1.0 and self.progress_stats.processed_files < self.progress_stats.total_files:
            self.progress_stats.processed_files += 1

        # Update estimated time remaining
        if progress > 0:
            elapsed_time = time.time() - self.start_time
            self.progress_stats.estimated_time_remaining = (
                elapsed_time / progress) * (1 - progress)

    def add_event(self, message: str, level: str = "INFO", details: Dict = None):
        """Add a new event to the log"""
        event = ProcessingEvent(
            timestamp=time.time(),
            event_type="PROCESS",
            message=message,
            level=level,
            details=details
        )
        self.events.append(event)
        self.analyzer.add_event(event)

    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, Enum):
            return obj.name
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._prepare_for_json(obj.__dict__)
        return obj

    def generate_final_report(self, output_dir: Path):
        """Generate final processing report"""
        # Create reports directory inside the specific output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate performance analysis
        analysis = self.analyzer.generate_performance_report()

        if not analysis:
            self.logger.warning("No performance data to generate report.")
            return

        # Add performance tracking for plots
        self._track_performance_data()

        # Generate plots with proper data tracking
        self._generate_plots(report_dir)

        # Convert system_info to be JSON serializable
        system_info = asdict(self.config_manager.system_info)
        system_info['system_type'] = system_info['system_type'].name

        # Create comprehensive report
        report = {
            "processing_summary": {
                "total_duration": float(analysis['processing_summary']['total_duration']),
                "total_processing_time": float(analysis['processing_summary']['total_processing_time']),
                "average_speed": float(analysis['processing_summary']['average_speed']),
                "files_processed": int(analysis['processing_summary']['files_processed'])
            },
            "resource_usage": {
                "peak_memory": float(analysis['resource_usage']['peak_memory']),
                "peak_gpu_memory": float(analysis['resource_usage']['peak_gpu_memory']),
                "average_cpu_usage": float(analysis['resource_usage']['average_cpu_usage']),
                "average_gpu_usage": float(analysis['resource_usage']['average_gpu_usage'])
            },
            "error_analysis": analysis['error_analysis'],
            "system_info": self._prepare_for_json(system_info),
            "event_log": [self._prepare_for_json(asdict(event)) for event in self.events],
            "plots_generated": [
                str(report_dir / "speed_plot.png"),
                str(report_dir / "resource_plot.png")
            ]
        }

        # Save report
        report_path = report_dir / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        # Generate HTML report
        self._generate_html_report(report, report_dir)

        return report_path

    def _track_performance_data(self):
        """Track performance data for plotting"""
        if not hasattr(self, 'performance_data'):
            self.performance_data = {
                'timestamps': [],
                'speeds': [],
                'memory_usage': [],
                'gpu_memory_usage': []
            }

        current_time = time.time()
        if hasattr(self, 'start_time'):
            elapsed = current_time - self.start_time

            # Calculate current speed
            if hasattr(self.progress_stats, 'total_processed_duration'):
                speed = getattr(self.progress_stats, 'total_processed_duration', 0) / elapsed
                self.performance_data['speeds'].append(speed)
                self.performance_data['timestamps'].append(elapsed)

            # Track memory usage
            memory = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
            self.performance_data['memory_usage'].append(memory)

            # Track GPU memory if available
            try:
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    self.performance_data['gpu_memory_usage'].append(gpu_memory)
                else:
                    self.performance_data['gpu_memory_usage'].append(0)
            except Exception as e:
                self.logger.debug(f"Error tracking GPU memory: {e}")
                self.performance_data['gpu_memory_usage'].append(0)

    def _generate_plots(self, report_dir: Path):
        """Generate performance visualization plots"""
        if not hasattr(self, 'performance_data') or not self.performance_data['timestamps']:
            self.logger.warning("No performance data available for plotting")
            return

        # Processing Speed Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance_data['timestamps'], self.performance_data['speeds'])
        plt.title('Processing Speed Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (x)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(report_dir / 'speed_plot.png')
        plt.close()

        # Resource Usage Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance_data['timestamps'],
                self.performance_data['memory_usage'],
                label='Memory (GB)', color='blue')

        if any(x > 0 for x in self.performance_data['gpu_memory_usage']):
            plt.plot(self.performance_data['timestamps'],
                    self.performance_data['gpu_memory_usage'],
                    label='GPU Memory (GB)', color='red')

        plt.title('Resource Usage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (GB)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_dir / 'resource_plot.png')
        plt.close()

    def _generate_html_report(self, report: Dict, report_dir: Path):
            """Generate HTML version of the report"""
            html_content = f"""
            <html>
            <head>
                <title>Transcription Processing Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
                    .plot {{ margin: 20px 0; }}
                    .error {{ color: red; }}
                    .warning {{ color: orange; }}
                    .success {{ color: green; }}
                </style>
            </head>
            <body>
                <h1>Transcription Processing Report</h1>

                <div class="section">
                    <h2>Processing Summary</h2>
                    <p>Total Duration: {timedelta(seconds=int(report['processing_summary']['total_duration']))}</p>
                    <p>Average Speed: {report['processing_summary']['average_speed']:.2f}x</p>
                    <p>Files Processed: {report['processing_summary']['files_processed']}</p>
                </div>

                <div class="section">
                    <h2>Resource Usage</h2>
                    <p>Peak Memory: {humanize.naturalsize(report['resource_usage']['peak_memory'])}</p>
                    <p>Peak GPU Memory: {humanize.naturalsize(report['resource_usage']['peak_gpu_memory'])}</p>
                </div>

                <div class="section">
                    <h2>Performance Plots</h2>
                    <div class="plot">
                        <img src="speed_plot.png" alt="Speed Plot">
                    </div>
                    <div class="plot">
                        <img src="resource_plot.png" alt="Resource Usage Plot">
                    </div>
                </div>

                <div class="section">
                    <h2>Error Analysis</h2>
                    <p>Total Errors: {report['error_analysis']['total_errors']}</p>
                    <p>Error Rate: {report['error_analysis']['error_rate']:.2%}</p>
                </div>

                <div class="section">
                    <h2>Event Log</h2>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Time</th>
                            <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Level</th>
                            <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Message</th>
                        </tr>
                        {"".join(
                            f'''
                            <tr class="{event['level'].lower()}">
                                <td style="padding: 8px; border: 1px solid #ddd;">
                                    {datetime.fromtimestamp(event['timestamp']).strftime("%H:%M:%S")}
                                </td>
                                <td style="padding: 8px; border: 1px solid #ddd;">{event['level']}</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">{event['message']}</td>
                            </tr>
                            '''
                            for event in report['event_log']
                        )}
                    </table>
                </div>
            </body>
            </html>
            """

            # Save HTML report
            html_path = report_dir / "report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
