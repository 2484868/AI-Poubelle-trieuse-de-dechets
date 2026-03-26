import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ The building block of ResNet """
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Network(nn.Module):
    def __init__(self, device: str, num_classes=6):
        super().__init__()
        self.device = device
        # This tracker must start at 64 because that's the output of our first conv layer
        self.in_planes = 64
        self.scaler = torch.amp.GradScaler('cuda' if 'cuda' in str(device) else 'cpu')

        # --- PART 1 : RESNET STEM ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # --- PART 2 : RESNET LAYERS ---
        # Fixed: Removed the redundant first argument that caused the 'multiple values' error
        self.layer1 = self._make_layer(64,  num_blocks=2, stride=1)   # Output: 64 channels, 128x128
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)  # Output: 128 channels, 64x64
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)  # Output: 256 channels, 32x32
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)  # Output: 512 channels, 16x16
        
        # --- PART 3 : ADAPTIVE POOLING & CLASSIFIER ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.linear_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        self.to(device)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear_layer(out)
        return out

    def train_model(self, dataloader, loss_fn, optimizer):
        self.train()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda' if 'cuda' in str(self.device) else 'cpu'):
                pred = self(X)
                loss = loss_fn(pred, y)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            current_processed = (batch + 1) * len(X)
            percent = 100 * ((batch + 1) / num_batches)
            bar = "#" * int(percent / 5) + "-" * (20 - int(percent / 5))
            
            print(f"\rTraining: [{bar}] {percent:>3.0f}% | Batch: {batch+1}/{num_batches} | Loss: {loss.item():>7f}", end="")

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
        return avg_loss