from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # ou yolov8s.pt pour plus de précision

results = model.train(
    data='dataset/data.yaml',
    epochs=150,          # Ultralytics recommande 300, 150 suffit pour peu de classes
    imgsz=640,
    batch=16,            # Meilleure généralisation que batch=8
    name='poubelle_v7',
    device=0,            # GPU (mettre 'cpu' si pas de GPU)
    patience=30,         # Early stopping si pas d'amélioration après 30 epochs
    augment=True,        # Augmentation de données intégrée
    val=True,            # Validation automatique
    plots=True           # Génère les courbes mAP, loss, etc.
)
