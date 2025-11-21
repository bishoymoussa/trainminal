#!/usr/bin/env python3
"""Quick test script to verify monitor is working."""

from trainminal import Monitor
from trainminal.metrics import Phase
import time

print("Starting monitor test...")
print("You should see the phase change from IDLE -> TRAINING -> VALIDATION")

with Monitor() as mon:
    # Wait a moment to see initial IDLE state
    print("Initial state - should show IDLE")
    time.sleep(1)
    
    # Set to TRAINING
    print("Setting phase to TRAINING...")
    mon.set_phase(Phase.TRAINING)
    mon.set_epoch(0, 5)
    mon.set_learning_rate(0.001)
    mon.log_metric('loss', 0.5)
    mon.log_metric('accuracy', 85.0)
    time.sleep(2)
    
    # Set to VALIDATION
    print("Setting phase to VALIDATION...")
    mon.set_phase(Phase.VALIDATION)
    mon.log_metric('val_loss', 0.4)
    mon.log_metric('val_accuracy', 87.0)
    time.sleep(2)
    
    print("Test complete!")

