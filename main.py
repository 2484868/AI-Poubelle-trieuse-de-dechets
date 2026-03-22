import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from network.with_pytorch.network import Network

def crop_black(image):
    gray = np.array(image.convert("L")) / 255.0
    mask = gray < 0.15

    coords = np.argwhere(mask)
    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return image.crop((x_min, y_min, x_max + 1, y_max + 1))

if __name__ == "__main__":
    default_transform = torchvision.transforms.Compose([
        transforms.Lambda(crop_black),
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Load and Split (Ensures different transforms for train/test)
    full_dataset = torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=default_transform)
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8 * len(indices))]
    test_idx = indices[int(0.8 * len(indices)):]

    # compute mean and std
    loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    mean = 0.
    std = 0.
    total_pixels = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)

        mean += images.sum(2).sum(0)
        std += (images ** 2).sum(2).sum(0)
        total_pixels += images.size(0) * images.size(2)

    mean /= total_pixels
    std = (std / total_pixels - mean ** 2).sqrt()
    # UNDERSTAND THE CODE ABOVE ?!?!?!

    # Training transforms: Includes Augmentation
    train_transform = torchvision.transforms.Compose([
        transforms.Lambda(crop_black),
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    print(f"mean: {mean.tolist()}, std: {std.tolist()}")

    # Testing transforms: Clean images for honest accuracy
    test_transform = torchvision.transforms.Compose([
        transforms.Lambda(crop_black),
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    train_data = Subset(torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=train_transform), train_idx)
    test_data = Subset(torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=test_transform), test_idx)

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Model Setup
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = Network(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    # Using SGD with momentum as requested
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # The Scheduler (Watches test_loss to adjust LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training Loop
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.train_model(train_dataloader, loss_fn, optimizer)
        
        # This now captures the average loss returned by the network
        current_test_loss = model.test_model(test_dataloader, loss_fn)
        
        # Update the learning rate based on performance
        scheduler.step(current_test_loss)
        
        print(f"Current LR: {optimizer.param_groups[0]['lr']}")

    print("Done!")
    torch.save(model.state_dict(), "network/saved_models/new_model.pth")