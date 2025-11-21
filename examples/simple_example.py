"""Simple example showing basic trainminal usage."""

import time
import random
from trainminal import Monitor
from trainminal.metrics import Phase


def simple_training_example():
    """Simple example of using trainminal for monitoring."""
    
    # Create monitor - can be used as context manager
    with Monitor() as mon:
        # Set training parameters
        num_epochs = 5
        batches_per_epoch = 10
        
        # Set initial phase to TRAINING before starting
        mon.set_phase(Phase.TRAINING)
        mon.set_epoch(0, num_epochs)
        mon.set_learning_rate(0.001)
        
        # Give display a moment to update
        time.sleep(0.1)
        
        for epoch in range(num_epochs):
            # Training phase
            mon.set_phase(Phase.TRAINING)
            mon.set_epoch(epoch, num_epochs)
            
            for batch in range(batches_per_epoch):
                mon.set_batch(batch, batches_per_epoch)
                
                # Simulate training
                time.sleep(0.1)
                
                # Simulate metrics
                loss = 1.0 - (epoch * 0.15 + batch * 0.01) + random.uniform(-0.05, 0.05)
                accuracy = (epoch * 10 + batch) + random.uniform(-2, 2)
                
                # Log metrics
                mon.log_metric('loss', max(0.1, loss))
                mon.log_metric('accuracy', min(100, max(0, accuracy)))
                mon.update_performance(batch_size=32)
            
            # Validation phase
            mon.set_phase(Phase.VALIDATION)
            time.sleep(0.5)
            
            val_loss = 1.0 - (epoch * 0.12) + random.uniform(-0.1, 0.1)
            val_accuracy = (epoch * 12) + random.uniform(-3, 3)
            
            mon.log_metric('val_loss', max(0.1, val_loss))
            mon.log_metric('val_accuracy', min(100, max(0, val_accuracy)))


def decorator_example():
    """Example using trainminal as a decorator."""
    
    from trainminal import monitor
    from trainminal.metrics import Phase
    
    @monitor()
    def my_training_function():
        mon = None  # In decorator mode, you'd need to get the monitor instance
        # For now, use context manager approach
        pass


if __name__ == '__main__':
    print("Running simple training example with trainminal...")
    simple_training_example()
    print("\nTraining complete!")

