"""Test script for ASCII plotting."""

from trainminal import Monitor
from trainminal.metrics import Phase
import time
import random


def test_plotting():
    """Test ASCII plotting functionality."""
    
    with Monitor() as mon:
        mon.set_phase(Phase.TRAINING)
        mon.set_epoch(0, 10)
        
        # Simulate training with metrics that will be plotted
        for step in range(50):
            # Simulate decreasing loss
            loss = 1.0 - (step * 0.015) + random.uniform(-0.05, 0.05)
            accuracy = (step * 1.5) + random.uniform(-2, 2)
            
            mon.log_metric('loss', max(0.1, loss))
            mon.log_metric('accuracy', min(100, max(0, accuracy)))
            
            if step % 10 == 0:
                # Validation
                mon.set_phase(Phase.VALIDATION)
                val_loss = loss * 0.9 + random.uniform(-0.05, 0.05)
                val_accuracy = accuracy + random.uniform(-1, 1)
                mon.log_metric('val_loss', max(0.1, val_loss))
                mon.log_metric('val_accuracy', min(100, max(0, val_accuracy)))
                mon.set_phase(Phase.TRAINING)
            
            time.sleep(0.1)
        
        # Show plots at the end
        print("\n" + "=" * 80)
        print("Loss Plot:")
        print("=" * 80)
        print(mon.get_plot('loss'))
        print("\n" + "=" * 80)
        print("Accuracy Plot:")
        print("=" * 80)
        print(mon.get_plot('accuracy'))
        print("\n" + "=" * 80)
        print("Combined Plots:")
        print("=" * 80)
        print(mon.get_plots(['loss', 'accuracy']))


if __name__ == '__main__':
    test_plotting()

