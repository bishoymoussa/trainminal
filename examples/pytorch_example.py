"""Example PyTorch training script with trainminal."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from trainminal import Monitor
from trainminal.metrics import Phase

# Create a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_with_monitor():
    """Train a model with trainminal monitoring."""
    
    # Create monitor
    with Monitor() as monitor:
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        train_data = torch.randn(1000, 28, 28)
        train_labels = torch.randint(0, 10, (1000,))
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_data = torch.randn(200, 28, 28)
        val_labels = torch.randint(0, 10, (200,))
        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        num_epochs = 10
        monitor.set_epoch(0, num_epochs)
        monitor.set_learning_rate(0.001)
        
        for epoch in range(num_epochs):
            # Training phase
            monitor.set_phase(Phase.TRAINING)
            monitor.set_epoch(epoch, num_epochs)
            model.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                monitor.set_batch(batch_idx, len(train_loader))
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                # Log metrics
                monitor.log_metric('loss', loss.item())
                monitor.log_metric('accuracy', 100. * train_correct / train_total)
                monitor.update_performance(batch_size=data.size(0))
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            monitor.set_phase(Phase.VALIDATION)
            model.eval()
            
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Log epoch metrics
            monitor.log_metric('train_loss', avg_train_loss)
            monitor.log_metric('train_accuracy', train_acc)
            monitor.log_metric('val_loss', avg_val_loss)
            monitor.log_metric('val_accuracy', val_acc)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")


if __name__ == '__main__':
    train_with_monitor()

