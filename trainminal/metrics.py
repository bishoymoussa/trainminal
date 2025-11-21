"""Metrics collection and storage for training monitoring."""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    """Training phases."""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    IDLE = "idle"


@dataclass
class MetricEntry:
    """Single metric entry."""
    name: str
    value: float
    step: int
    phase: Phase
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collect and store training metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.current_phase = Phase.IDLE
        self.current_step = 0
        self.current_epoch = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Training state
        self.total_epochs = None
        self.current_batch = 0
        self.total_batches = None
        self.learning_rate = None
        
        # Performance metrics
        self.samples_per_second = 0.0
        self.batches_per_second = 0.0
        self.last_batch_time = None
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if step is None:
            step = self.current_step
        
        entry = MetricEntry(
            name=name,
            value=value,
            step=step,
            phase=self.current_phase,
            timestamp=time.time()
        )
        
        self.metrics[name].append(entry)
        self.last_update_time = time.time()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def set_phase(self, phase: Phase):
        """Set the current training phase."""
        self.current_phase = phase
    
    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None):
        """Set the current epoch."""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
        self.current_step = epoch  # Use epoch as step if not explicitly set
    
    def set_batch(self, batch: int, total_batches: Optional[int] = None):
        """Set the current batch."""
        self.current_batch = batch
        if total_batches is not None:
            self.total_batches = total_batches
    
    def set_learning_rate(self, lr: float):
        """Set the current learning rate."""
        self.learning_rate = lr
    
    def update_performance(self, batch_size: int, batch_time: Optional[float] = None):
        """Update performance metrics based on batch processing."""
        current_time = time.time()
        
        if batch_time is None:
            if self.last_batch_time is not None:
                batch_time = current_time - self.last_batch_time
            else:
                batch_time = 0.0
        
        if batch_time > 0:
            self.batches_per_second = 1.0 / batch_time
            self.samples_per_second = batch_size / batch_time
        
        self.last_batch_time = current_time
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest value for each metric."""
        latest = {}
        for name, entries in self.metrics.items():
            if entries:
                latest[name] = entries[-1].value
        return latest
    
    def get_metric_history(self, name: str) -> List[MetricEntry]:
        """Get the full history for a specific metric."""
        return list(self.metrics.get(name, []))
    
    def get_phase_metrics(self, phase: Phase) -> Dict[str, List[MetricEntry]]:
        """Get all metrics for a specific phase."""
        phase_metrics = defaultdict(list)
        for name, entries in self.metrics.items():
            for entry in entries:
                if entry.phase == phase:
                    phase_metrics[name].append(entry)
        return dict(phase_metrics)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        epoch_progress = None
        if self.total_epochs is not None:
            epoch_progress = (self.current_epoch + 1) / self.total_epochs
        
        batch_progress = None
        if self.total_batches is not None:
            batch_progress = (self.current_batch + 1) / self.total_batches
        
        elapsed = time.time() - self.start_time
        
        return {
            'epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'epoch_progress': epoch_progress,
            'batch': self.current_batch,
            'total_batches': self.total_batches,
            'batch_progress': batch_progress,
            'phase': self.current_phase.value,
            'elapsed': elapsed,
            'learning_rate': self.learning_rate,
            'samples_per_second': self.samples_per_second,
            'batches_per_second': self.batches_per_second,
        }
    
    def check_anomalies(self) -> List[str]:
        """Check for anomalies in metrics (NaN, inf, etc.)."""
        anomalies = []
        
        for name, entries in self.metrics.items():
            if entries:
                latest = entries[-1].value
                if isinstance(latest, float):
                    import math
                    if math.isnan(latest):
                        anomalies.append(f"Metric '{name}' is NaN")
                    elif math.isinf(latest):
                        anomalies.append(f"Metric '{name}' is infinite")
        
        return anomalies
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.current_phase = Phase.IDLE
        self.current_step = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.learning_rate = None
        self.samples_per_second = 0.0
        self.batches_per_second = 0.0
        self.last_batch_time = None

