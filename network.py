from torch import nn
import torch

class Network(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 16x16
        )
        
        self.flatten = nn.Flatten()
        
        self.linear_layer = nn.Sequential(
            nn.Linear(32*14*14, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
        )
        self.to(device)
    
    def forward(self, x):
        x = self.convolutional_layer(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x

    def train_model(self, dataloader, loss_fn, optimizer):
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            pred = self(X)
            loss = loss_fn(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_model(self, dataloader, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval() 
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        avg_loss = test_loss / num_batches
        accuracy = 100 * (correct / size)
        
        print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_loss:>8f}")
        
        # --- FIXED: Returning the loss for the scheduler ---
        return avg_loss