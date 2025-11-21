"""Test multi-GPU display."""

from trainminal import Monitor
from trainminal.metrics import Phase
import time

# Test that all GPUs are displayed
with Monitor() as mon:
    mon.set_phase(Phase.TRAINING)
    mon.set_epoch(0, 5)
    
    # Log some metrics
    for i in range(10):
        mon.log_metric('loss', 1.0 - i * 0.1)
        mon.log_metric('accuracy', i * 10)
        time.sleep(0.2)
    
    print("\nMulti-GPU test complete!")
    print("Check the display above - all GPUs should be shown.")

