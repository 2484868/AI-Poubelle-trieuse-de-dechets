from ultralytics import YOLO

model = YOLO("best.pt")  # chemin local au lieu de Hugging Face

results = model(r"C:\Users\matis\OneDrive\1-Etude\0_Projet_Integration\AI_Poubelle_Intelligente\dataset_2\dataset_a_labelliser\recyclage\papier_20260129_180700.jpg")  # chemin local de l'image à tester

results[0].show()
