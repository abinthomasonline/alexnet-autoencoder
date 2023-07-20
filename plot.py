"""Plot metrics and images"""

import json
import os

import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import Autoencoder


def plot(*args, **kwargs):

    # Configs
    checkpoint_dir = "checkpoints"
    plots_dir = "plots"
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    num_images = 10
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")

    # Load metrics
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Plot metrics
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics["epoch"], metrics["train"], label="Train")
    ax.plot(metrics["epoch"], metrics["eval"], label="Eval")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MSE Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss.png"))

    # Data
    image_path = "data/imagenette2"
    val_path = os.path.join(image_path, "val")
    val_dataset = datasets.ImageFolder(
        val_path, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=num_images, shuffle=True)
    images, _ = next(iter(val_loader))

    # Model
    model = Autoencoder()
    inverse_transform = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    # Original images
    original_images = torch.clamp(inverse_transform(images), 0, 1)

    # Best checkpoint
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        best_reconstructions = model(images)
    best_reconstructions = torch.clamp(inverse_transform(best_reconstructions), 0, 1)

    # checkpoints
    intermediate_reconstructions = []
    epochs = sorted([int(filename.split(".")[0].split("_")[1])
                     for filename in os.listdir(checkpoint_dir)
                     if filename.startswith("epoch_") and filename.endswith(".pt")], reverse=True)
    for epoch in epochs:
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            reconstructions = model(images)
        reconstructions = torch.clamp(inverse_transform(reconstructions), 0, 1)
        intermediate_reconstructions.append(reconstructions)

    # Plot
    n_rows = num_images
    n_cols = len([original_images, best_reconstructions] + intermediate_reconstructions)
    img_size = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*img_size, n_rows*img_size))
    for i in range(n_rows):
        # Original images
        ax[i][0].imshow(torch.permute(original_images[i], (1, 2, 0)))
        ax[i][0].get_xaxis().set_visible(False)
        ax[i][0].get_yaxis().set_visible(False)
        ax[i][0].set_title("original")

        # Best reconstructions
        ax[i][1].imshow(torch.permute(best_reconstructions[i], (1, 2, 0)))
        ax[i][1].get_xaxis().set_visible(False)
        ax[i][1].get_yaxis().set_visible(False)
        ax[i][1].set_title("best")

        # Intermediate reconstructions
        for j, epoch in enumerate(epochs):
            intermediate_reconstruction = intermediate_reconstructions[j]
            ax[i][j + 2].imshow(torch.permute(intermediate_reconstruction[i], (1, 2, 0)))
            ax[i][j + 2].get_xaxis().set_visible(False)
            ax[i][j + 2].get_yaxis().set_visible(False)
            ax[i][j + 2].set_title(f"epoch {epoch}")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "reconstructions.png"))


if __name__ == "__main__":
    plot()
