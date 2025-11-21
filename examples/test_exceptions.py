"""Test script for exception handling."""

from trainminal import Monitor
from trainminal.metrics import Phase
import time


def test_exception_handling():
    """Test that exceptions are displayed cleanly."""
    
    with Monitor() as mon:
        mon.set_phase(Phase.TRAINING)
        mon.set_epoch(0, 5)
        mon.log_metric('loss', 0.5)
        
        print("Running for 2 seconds, then raising an exception...")
        time.sleep(2)
        
        # Raise an exception to test handling
        raise ValueError("This is a test exception to verify clean exception display!")


if __name__ == '__main__':
    try:
        test_exception_handling()
    except Exception:
        # Exception should be handled by monitor
        pass

