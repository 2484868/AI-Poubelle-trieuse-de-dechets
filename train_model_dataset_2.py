#!/usr/bin/env python3
"""
Script d'entraînement avec correctif PyTorch 2.6
"""
import torch

# --- CORRECTIF PYTORCH 2.6 ---
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load
# --- FIN CORRECTIF ---

from ultralytics import YOLO

if __name__ == '__main__':
    # Charge le modèle de base
    model = YOLO('yolov8n.pt')
    
    # Lance l'entraînement
    results = model.train(
        data='dataset_final/data.yaml',
        epochs=10, # Variable qui représente le nombre de tentative d'algorithme et va garder le meilleur d'entre eux
        imgsz=640,
        batch=4,  # Réduis si tu manques de RAM (4 ou 2 sur RPi)
        patience=20,
        name='poubelle_v7'  # Nom du dossier de résultats
    )
    
    print("\n✅ Entraînement terminé !")
    print(f"Modèle sauvegardé dans : runs/detect/poubelle_v1/weights/best.pt")
