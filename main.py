import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.optim as optim
import numpy as np
from network import Network
import os 
import kagglehub

def crop_black(image):
    gray = np.array(image.convert("L")) / 255.0
    mask = gray < 0.05

    coords = np.argwhere(mask)
    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return image.crop((x_min, y_min, x_max + 1, y_max + 1))

if __name__ == "__main__":
    default_transform = v2.Compose([
        v2.Lambda(crop_black),
        v2.Resize((128, 128)),
        v2.ToTensor(),
    ])
    raw_path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
    data_dir = os.path.join(raw_path, 'Garbage classification', 'Garbage classification')

    full_dataset = torchvision.datasets.ImageFolder(
        root=data_dir, 
        transform=default_transform
    )
    
    # Now your main script "knows" exactly where the 2,527 images are!

    # Load and Split (Ensures different transforms for train/test)
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8 * len(indices))]
    test_idx = indices[int(0.8 * len(indices)):]

    # compute mean and std
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    mean = 0.
    std = 0.
    total_pixels = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)

        mean += images.sum(2).sum(0)
        std += (images ** 2).sum(2).sum(0)
        total_pixels += images.size(0) * images.size(2)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # UNDERSTAND THE CODE ABOVE ?!?!?!

    # Training transforms: Includes Augmentation
    train_transform = v2.Compose([
        v2.Lambda(crop_black),
        v2.Resize((128, 128)),
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    # Testing transforms: Clean images for honest accuracy
    test_transform = v2.Compose([
        v2.Lambda(crop_black),
        v2.Resize((128, 128)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    train_data = Subset(torchvision.datasets.ImageFolder(root=data_dir, transform=train_transform), train_idx)
    test_data = Subset(torchvision.datasets.ImageFolder(root=data_dir, transform=test_transform), test_idx)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = Network(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    # Using SGD with momentum as requested
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # The Scheduler (Watches test_loss to adjust LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Training Loop
    epochs = 20    ## A CHANGER##########
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.train_model(train_dataloader, loss_fn, optimizer)
        
        # This now captures the average loss returned by the network
        current_test_loss = model.test_model(test_dataloader, loss_fn)
        
        # Update the learning rate based on performance
        scheduler.step(current_test_loss)
        
        print(f"Current LR: {optimizer.param_groups[0]['lr']}")

    print("Done!")
    torch.save(model.state_dict(), "saved_models/new_model.pth")