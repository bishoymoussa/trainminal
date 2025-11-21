"""Main Monitor class for training monitoring."""

import sys
import os
import signal
import threading
import time
import json
from typing import Optional, Callable, Any
from contextlib import contextmanager
from functools import wraps

from trainminal.metrics import MetricsCollector, Phase
from trainminal.display import DisplayManager
from trainminal.utils import detect_framework
from trainminal.exceptions import ExceptionHandler
from trainminal.plotting import ASCIIPlotter


class Monitor:
    """Main monitoring class for ML/DL training."""
    
    def __init__(
        self,
        use_tui: bool = True,
        refresh_rate: float = 0.5,
        log_file: Optional[str] = None,
        auto_detect: bool = True,
        enable_plotting: bool = True,
        enable_exception_handling: bool = True
    ):
        self.metrics = MetricsCollector()
        self.display = DisplayManager(use_tui=use_tui, refresh_rate=refresh_rate)
        self.log_file = log_file
        self.auto_detect = auto_detect
        self.framework = None
        self.enable_plotting = enable_plotting
        self.enable_exception_handling = enable_exception_handling
        
        # Exception handling
        if enable_exception_handling:
            self.exception_handler = ExceptionHandler(console=self.display.console)
        else:
            self.exception_handler = None
        
        # Plotting
        if enable_plotting:
            self.plotter = ASCIIPlotter()
        else:
            self.plotter = None
        
        self._running = False
        self._display_thread = None
        self._shutdown_requested = False
        
        # Check if running under trainminal CLI
        self.shared_state_path = os.environ.get('TRAINMINAL_SHARED_STATE')
        self.cli_mode = os.environ.get('TRAINMINAL_CLI_MODE') == '1'
        
        # If in CLI mode, disable TUI (CLI wrapper will handle it)
        if self.cli_mode and use_tui:
            use_tui = False
        
        # Recreate display if we changed use_tui
        if self.cli_mode:
            self.display = DisplayManager(use_tui=False, refresh_rate=refresh_rate)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        if auto_detect:
            self.framework = detect_framework()
            if self.framework:
                self._setup_framework_hooks()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self._shutdown_requested = True
        self.stop()
        sys.exit(0)
    
    def _setup_framework_hooks(self):
        """Setup automatic hooks for detected framework."""
        if self.framework == "pytorch":
            self._setup_pytorch_hooks()
        elif self.framework == "tensorflow":
            self._setup_tensorflow_hooks()
        elif self.framework == "jax":
            self._setup_jax_hooks()
    
    def _setup_pytorch_hooks(self):
        """Setup hooks for PyTorch training."""
        try:
            import torch
            import torch.nn as nn
            
            # Hook into backward pass to detect training
            original_backward = torch.Tensor.backward
            
            def hooked_backward(self, *args, **kwargs):
                if not self._shutdown_requested:
                    self.set_phase(Phase.TRAINING)
                return original_backward(self, *args, **kwargs)
            
            # Note: This is a simple hook, more sophisticated hooks would
            # require integration with training loops directly
        except ImportError:
            pass
    
    def _setup_tensorflow_hooks(self):
        """Setup hooks for TensorFlow training."""
        # TensorFlow hooks would go here
        pass
    
    def _setup_jax_hooks(self):
        """Setup hooks for JAX training."""
        # JAX hooks would go here
        pass
    
    def _display_loop(self):
        """Main display update loop."""
        while self._running and not self._shutdown_requested:
            try:
                self.display.update(self.metrics, self.plotter)
                time.sleep(self.display.refresh_rate)
            except Exception:
                break
    
    def start(self):
        """Start the monitoring system."""
        if self._running:
            return
        
        self._running = True
        self.display.start()
        
        # Start display update thread
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
    
    def stop(self):
        """Stop the monitoring system."""
        if not self._running:
            return
        
        self._running = False
        self.display.stop()
        
        if self._display_thread:
            self._display_thread.join(timeout=1.0)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        self.metrics.log_metric(name, value, step)
        # Add to plotter if enabled
        if self.plotter:
            self.plotter.add_point(name, value)
        # Write to shared state if in CLI mode
        if self.cli_mode and self.shared_state_path:
            self._write_shared_state()
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log multiple metrics at once."""
        self.metrics.log_metrics(metrics, step)
    
    def set_phase(self, phase: Phase):
        """Set the current training phase."""
        self.metrics.set_phase(phase)
        # Write to shared state if in CLI mode
        if self.cli_mode and self.shared_state_path:
            self._write_shared_state()
    
    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None):
        """Set the current epoch."""
        self.metrics.set_epoch(epoch, total_epochs)
        # Write to shared state if in CLI mode
        if self.cli_mode and self.shared_state_path:
            self._write_shared_state()
    
    def set_batch(self, batch: int, total_batches: Optional[int] = None):
        """Set the current batch."""
        self.metrics.set_batch(batch, total_batches)
        # Write to shared state if in CLI mode
        if self.cli_mode and self.shared_state_path:
            self._write_shared_state()
    
    def set_learning_rate(self, lr: float):
        """Set the current learning rate."""
        self.metrics.set_learning_rate(lr)
        # Write to shared state if in CLI mode
        if self.cli_mode and self.shared_state_path:
            self._write_shared_state()
    
    def update_performance(self, batch_size: int, batch_time: Optional[float] = None):
        """Update performance metrics."""
        self.metrics.update_performance(batch_size, batch_time)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        
        # Handle exceptions if they occurred
        if exc_type is not None and self.exception_handler:
            self.exception_handler.display_exception(exc_type, exc_val, exc_tb)
        
        self.display.cleanup()
        return False  # Don't suppress exceptions
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator support."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with self:
                    return func(*args, **kwargs)
            except Exception as e:
                if self.exception_handler:
                    self.exception_handler.display_exception(
                        type(e), e, e.__traceback__
                    )
                raise
        return wrapper
    
    def get_plot(self, metric_name: str, title: Optional[str] = None) -> str:
        """Get an ASCII plot for a metric."""
        if not self.plotter:
            return "Plotting is disabled"
        return self.plotter.plot(metric_name, title)
    
    def get_plots(self, metric_names: list, title: Optional[str] = None) -> str:
        """Get multiple plots side by side."""
        if not self.plotter:
            return "Plotting is disabled"
        return self.plotter.plot_multiple(metric_names, title)
    
    def _write_shared_state(self):
        """Write current state to shared file for CLI wrapper."""
        if not self.shared_state_path:
            return
        
        try:
            latest_metrics = self.metrics.get_latest_metrics()
            progress = self.metrics.get_progress()
            
            state = {
                'metrics': latest_metrics,
                'phase': self.metrics.current_phase.name,
                'epoch': progress.get('epoch'),
                'total_epochs': progress.get('total_epochs'),
                'batch': progress.get('batch'),
                'total_batches': progress.get('total_batches'),
                'learning_rate': progress.get('learning_rate'),
            }
            
            # Write atomically
            temp_path = self.shared_state_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(state, f)
            os.replace(temp_path, self.shared_state_path)
        except Exception:
            # Silently fail - don't break training if shared state fails
            pass


# Convenience function for decorator usage
def monitor(
    use_tui: bool = True,
    refresh_rate: float = 0.5,
    log_file: Optional[str] = None,
    auto_detect: bool = True
) -> Monitor:
    """Create a monitor instance for decorator usage."""
    return Monitor(use_tui=use_tui, refresh_rate=refresh_rate, log_file=log_file, auto_detect=auto_detect)

