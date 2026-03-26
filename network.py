from torch import nn
import torch

class Network(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        
        # --- PARTIE 1 : RÉSEAU CONVOLUTIONNEL (Extraction de caractéristiques) ---
        self.convolutional_layer = nn.Sequential(
            # Bloc 1 : Détection de formes simples (bordures, couleurs)
            # Entrée : (3 canaux, 128x128) -> Sortie : (32 filtres, 64x64 après MaxPool)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), # <-- AJOUTÉ pour la stabilité
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Bloc 2 : Combinaison de formes (cercles, carrés)
            # Sortie : (64 filtres, 32x32 après MaxPool)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), # <-- AJOUTÉ
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Bloc 3 : Détection de textures complexes (métal, papier froissé)
            # Sortie : (128 filtres, 16x16 après MaxPool)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), # <-- AJOUTÉ
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloc 4 : Nouveau bloc pour une meilleure abstraction (reconnaissance d'objets)
            # Sortie : (256 filtres, 8x8 après MaxPool)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), # <-- AJOUTÉ
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # --- PARTIE 2 : RÉSEAU LINÉAIRE (Classification) ---
        self.flatten = nn.Flatten()
        
        # Recalcul de la taille d'entrée : 256 filtres * (128/2/2/2/2)^2 = 256 * 8 * 8
        self.linear_layer = nn.Sequential(
            nn.Linear(65536, 512), # <-- Taille corrigée
            nn.ReLU(),
            nn.Dropout(0.4), # <-- Augmenté à 0.4 pour contrer l'overfitting
            nn.Linear(512, 256), # <-- Nouvelle couche pour plus de finesse
            nn.ReLU(),
            nn.Linear(256, 6) # <-- 6 catégories (verre, métal, etc.)
        )
        self.to(device)
    
    def forward(self, x):
        # Passage dans les convolutions
        x = self.convolutional_layer(x)
        
        # Aplatissement
        x = self.flatten(x)
        
        # Classification
        x = self.linear_layer(x)
        return x

    # --- Les fonctions train_model et test_model restent inchangées ---
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
        
        # Retourne la perte pour le scheduler
        return avg_loss