"""Display system for terminal UI using rich library."""

import sys
import time
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich import box

from trainminal.metrics import MetricsCollector, Phase
from trainminal.resources import ResourceMonitor
from trainminal.utils import format_time, format_bytes, format_percentage, calculate_eta


class DisplayManager:
    """Manage the terminal display for training monitoring."""
    
    def __init__(self, use_tui: bool = True, refresh_rate: float = 0.5):
        self.console = Console()
        self.use_tui = use_tui and sys.stdout.isatty()
        self.refresh_rate = refresh_rate
        self.live = None
        self.resource_monitor = ResourceMonitor()
        
    def create_layout(self, metrics: MetricsCollector, plotter=None) -> Layout:
        """Create the main layout for the TUI."""
        layout = Layout()
        
        # Split into main content and status bar
        layout.split_column(
            Layout(name="main", ratio=9),
            Layout(name="status", size=1)
        )
        
        # Split main into metrics and resources (no plots)
        layout["main"].split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="resources", ratio=1)
        )
        
        # Create content for each section
        layout["metrics"].update(self._create_metrics_panel(metrics))
        layout["resources"].update(self._create_resources_panel())
        layout["status"].update(self._create_status_bar(metrics))
        
        return layout
    
    def _create_plots_panel(self, plotter) -> Panel:
        """Create a panel showing ASCII plots."""
        from trainminal.plotting import ASCIIPlotter
        
        if not plotter or not plotter.metric_history:
            return Panel("No plots available", title="Plots", border_style="blue")
        
        # Get metrics that have enough data points and variation
        metric_names = list(plotter.metric_history.keys())
        
        # Filter metrics: need at least 2 points and some variation
        plottable_metrics = []
        for metric_name in metric_names:
            history = plotter.metric_history[metric_name]
            if len(history) < 2:
                continue  # Need at least 2 points
            
            # Check if metric has variation (not constant)
            values = list(history)
            min_val = min(values)
            max_val = max(values)
            if min_val == max_val:
                continue  # Constant value, skip
            
            # Skip non-training metrics
            if metric_name in ['status']:
                continue
            
            plottable_metrics.append(metric_name)
        
        if not plottable_metrics:
            return Panel("No plottable metrics\n(need 2+ points with variation)", 
                        title="Plots", border_style="blue")
        
        # Limit to 2-3 most important metrics for display
        priority_metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        metrics_to_plot = []
        
        for metric in priority_metrics:
            if metric in plottable_metrics:
                metrics_to_plot.append(metric)
                if len(metrics_to_plot) >= 2:
                    break
        
        # If we don't have priority metrics, take the first plottable ones
        if not metrics_to_plot:
            metrics_to_plot = plottable_metrics[:2]
        
        if not metrics_to_plot:
            return Panel("No metrics to plot", title="Plots", border_style="blue")
        
        # Create plots
        if len(metrics_to_plot) == 1:
            plot_text = plotter.plot(metrics_to_plot[0])
        else:
            plot_text = plotter.plot_multiple(metrics_to_plot[:2])
        
        # Format as text for display
        from rich.text import Text
        plot_lines = plot_text.split('\n')
        plot_display = Text()
        for line in plot_lines:
            plot_display.append(line + "\n")
        
        return Panel(plot_display, title="Plots", border_style="blue")
    
    def _create_metrics_panel(self, metrics: MetricsCollector) -> Panel:
        """Create the metrics display panel."""
        progress = metrics.get_progress()
        latest = metrics.get_latest_metrics()
        
        # Status text
        phase = progress['phase'].upper()
        phase_color = {
            'TRAINING': 'green',
            'VALIDATION': 'yellow',
            'TESTING': 'blue',
            'IDLE': 'white'
        }.get(phase, 'white')
        
        status_text = f"[{phase_color}]{phase}[/{phase_color}]"
        if progress['total_epochs']:
            status_text += f" Epoch {progress['epoch'] + 1}/{progress['total_epochs']}"
        
        # Metrics table
        table = Table.grid(padding=(0, 2))
        table.add_row(Text("Status:", style="bold"), status_text)
        
        # Add latest metrics
        for name, value in sorted(latest.items()):
            if isinstance(value, float):
                table.add_row(f"{name.capitalize()}:", f"{value:.6f}")
            else:
                table.add_row(f"{name.capitalize()}:", str(value))
        
        # Learning rate
        if progress['learning_rate'] is not None:
            table.add_row("Learning Rate:", f"{progress['learning_rate']:.2e}")
        
        # Progress bar
        if progress['epoch_progress'] is not None:
            progress_bar = self._create_progress_bar(
                progress['epoch_progress'],
                f"Epoch {progress['epoch'] + 1}/{progress['total_epochs']}"
            )
            table.add_row("", "")
            table.add_row(progress_bar, "")
        
        if progress['batch_progress'] is not None:
            batch_bar = self._create_progress_bar(
                progress['batch_progress'],
                f"Batch {progress['batch'] + 1}/{progress['total_batches']}"
            )
            table.add_row(batch_bar, "")
        
        # Performance metrics
        if progress['samples_per_second'] > 0:
            table.add_row("", "")
            table.add_row("Speed:", f"{progress['samples_per_second']:.1f} samples/s")
            table.add_row("", f"{progress['batches_per_second']:.2f} batches/s")
        
        # Time information
        elapsed = format_time(progress['elapsed'])
        table.add_row("", "")
        table.add_row("Elapsed:", elapsed)
        
        if progress['epoch_progress'] and progress['epoch_progress'] < 1.0:
            eta = calculate_eta(progress['elapsed'], progress['epoch_progress'])
            if eta is not None:
                eta_str = format_time(eta)
                table.add_row("ETA:", eta_str)
        
        return Panel(table, title="Training Metrics", border_style="cyan")
    
    def _create_progress_bar(self, progress: float, label: str) -> str:
        """Create a text-based progress bar."""
        width = 30
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        percent = int(progress * 100)
        return f"{label}: [{bar}] {percent}%"
    
    def _create_resources_panel(self) -> Panel:
        """Create the resources display panel."""
        resources = self.resource_monitor.get_all_resources()
        
        table = Table.grid(padding=(0, 1))
        
        # GPU information
        if resources['gpus']:
            gpu_count = len(resources['gpus'])
            gpu_label = f"GPUs ({gpu_count}):" if gpu_count > 1 else "GPUs:"
            table.add_row(Text(gpu_label, style="bold"), "")
            
            for gpu in resources['gpus']:
                gpu_name = f"GPU {gpu['index']}: {gpu['name']}"
                util_bar = self._create_utilization_bar(gpu['utilization'] / 100.0)
                vram = f"{format_bytes(gpu['memory_used'])}/{format_bytes(gpu['memory_total'])}"
                vram_percent = f"{gpu['memory_used_percent']:.1f}%"
                
                table.add_row(f"  {gpu_name}", "")
                table.add_row(f"    Util: {util_bar} {gpu['utilization']}%", "")
                table.add_row(f"    VRAM: {vram} ({vram_percent})", "")
                
                if gpu['temperature'] is not None:
                    table.add_row(f"    Temp: {gpu['temperature']}°C", "")
                if gpu['power'] is not None:
                    table.add_row(f"    Power: {gpu['power']:.1f}W", "")
                
                # Add spacing between GPUs if multiple
                if gpu['index'] < len(resources['gpus']) - 1:
                    table.add_row("", "")
        else:
            table.add_row(Text("GPUs:", style="bold"), "No GPU detected")
        
        table.add_row("", "")
        
        # CPU information
        cpu = resources['cpu']
        cpu_bar = self._create_utilization_bar(cpu['percent'] / 100.0)
        table.add_row(Text("CPU:", style="bold"), f"{cpu_bar} {cpu['percent']:.1f}%")
        
        # RAM information
        ram = resources['ram']
        ram_bar = self._create_utilization_bar(ram['percent'] / 100.0)
        ram_used = format_bytes(ram['used'])
        ram_total = format_bytes(ram['total'])
        table.add_row(Text("RAM:", style="bold"), f"{ram_bar} {ram['percent']:.1f}%")
        table.add_row("", f"{ram_used}/{ram_total}")
        
        return Panel(table, title="System Resources", border_style="green")
    
    def _create_utilization_bar(self, utilization: float) -> str:
        """Create a utilization bar."""
        width = 20
        filled = int(utilization * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def _create_status_bar(self, metrics: MetricsCollector) -> Text:
        """Create the status bar at the bottom."""
        progress = metrics.get_progress()
        anomalies = metrics.check_anomalies()
        
        status_parts = []
        
        if anomalies:
            status_parts.append(f"[red]WARNING: Anomalies: {', '.join(anomalies)}[/red]")
        
        status_parts.append("[dim]Press Ctrl+C to stop[/dim]")
        
        return Text(" | ".join(status_parts), justify="left")
    
    def start(self):
        """Start the live display."""
        if self.use_tui:
            self.live = Live(console=self.console, refresh_per_second=1.0/self.refresh_rate)
            self.live.start()
    
    def update(self, metrics: MetricsCollector, plotter=None):
        """Update the display with current metrics."""
        if self.use_tui and self.live:
            layout = self.create_layout(metrics, plotter)
            self.live.update(layout)
        else:
            # Fallback to simple text output
            self._print_simple_update(metrics, plotter)
    
    def _print_simple_update(self, metrics: MetricsCollector, plotter=None):
        """Print a simple text update (fallback mode)."""
        progress = metrics.get_progress()
        latest = metrics.get_latest_metrics()
        
        print("\033[2J\033[H", end="")  # Clear screen
        print(f"=== Training Monitor ===")
        print(f"Phase: {progress['phase'].upper()}")
        if progress['total_epochs']:
            print(f"Epoch: {progress['epoch'] + 1}/{progress['total_epochs']}")
        
        print("\nMetrics:")
        for name, value in sorted(latest.items()):
            if isinstance(value, float):
                print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")
        
        if progress['learning_rate']:
            print(f"Learning Rate: {progress['learning_rate']:.2e}")
        
        # Show plots if available
        if plotter and plotter.metric_history:
            print("\nPlots:")
            for metric_name in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
                if metric_name in plotter.metric_history and len(plotter.metric_history[metric_name]) > 1:
                    print(f"\n{metric_name}:")
                    print(plotter.plot(metric_name))
                    break
        
        # Resources
        resources = self.resource_monitor.get_all_resources()
        print("\nResources:")
        for gpu in resources['gpus']:
            print(f"  GPU {gpu['index']}: {gpu['utilization']}% | "
                  f"VRAM: {format_bytes(gpu['memory_used'])}/{format_bytes(gpu['memory_total'])}")
        
        cpu = resources['cpu']
        ram = resources['ram']
        print(f"  CPU: {cpu['percent']:.1f}%")
        print(f"  RAM: {ram['percent']:.1f}% ({format_bytes(ram['used'])}/{format_bytes(ram['total'])})")
        
        elapsed = format_time(progress['elapsed'])
        print(f"\nElapsed: {elapsed}")
    
    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None
    
    def cleanup(self):
        """Clean up display resources."""
        self.stop()
        if self.resource_monitor:
            self.resource_monitor.cleanup()

