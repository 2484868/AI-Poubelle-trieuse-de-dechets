import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import v2
import numpy as np
from network import Network
import os
import kagglehub

# 🔥 Speed boost for GPU
torch.backends.cudnn.benchmark = True

#maybe put back a more effecient black mask that can run faster and add a v2.lambda(defblackmask) to transform parameters



if __name__ == "__main__":

    # --- Load dataset path ---
    raw_path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
    data_dir = os.path.join(raw_path, 'Garbage classification', 'Garbage classification')

    # --- Base dataset (no transforms yet) ---
    full_dataset = torchvision.datasets.ImageFolder(root=data_dir)

    # --- Train / Test split ---
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8 * len(indices))]
    test_idx = indices[int(0.8 * len(indices)):]

    # --- Normalization (ImageNet stats) ---
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # --- Transforms ---
    train_transform = v2.Compose([
        v2.Resize((128, 128)),
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    test_transform = v2.Compose([
        v2.Resize((128, 128)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    # --- Datasets with transforms ---
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=test_transform
    )

    # --- Apply subsets ---
    train_data = Subset(train_dataset, train_idx)
    test_data = Subset(test_dataset, test_idx)

    # --- DataLoaders ---
    train_dataloader = DataLoader(
        train_data,
        batch_size=64,   # 🔥 increase to 128 if GPU allows
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model ---
    model = Network(device)

    # --- Loss & Optimizer ---
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2
    )

    # --- Training Loop ---
    epochs = 20
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")

        model.train_model(train_dataloader, loss_fn, optimizer)

        current_test_loss = model.test_model(test_dataloader, loss_fn)

        scheduler.step(current_test_loss)

        print(f"Current LR: {optimizer.param_groups[0]['lr']}")

    print("Done!")

    # --- Save model ---
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/new_model.pth")
