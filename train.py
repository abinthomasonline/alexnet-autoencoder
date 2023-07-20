"""Train the autoencoder model on the Imagenette dataset"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import Autoencoder


def train(*args, **kwargs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Configs and Hyperparameters
    image_path = "data/imagenette2"
    batch_size = 64
    learning_rate = 1e-3
    # momentum = 0.9
    epochs = 1000
    steps_per_epoch = 100
    val_steps = 100
    early_stopping_patience = 10
    checkpoint_dir = "checkpoints"
    os.mkdir(checkpoint_dir)
    checkpoint_interval = 10  # epochs
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")

    # Data
    train_path = os.path.join(image_path, "train")
    val_path = os.path.join(image_path, "val")

    train_dataset = datasets.ImageFolder(
        train_path, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))
    val_dataset = datasets.ImageFolder(
        val_path, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model, Loss, Optimizer
    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize training variables
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    metrics = {"epoch": [], "train": [], "eval": []}
    epoch = -1
    best_val_loss = np.inf
    early_stopping_counter = 0

    # Load from checkpoint if exists
    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] - 1
        early_stopping_counter = checkpoint["early_stopping_counter"]

    # Save zeroth epoch checkpoint for plotting
    if epoch == -1:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict()
        }, os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt"))

    # Training loop
    while (epoch + 1) < epochs:
        epoch += 1

        # Early stopping
        if early_stopping_counter == early_stopping_patience:
            print("Stopping Criterion Met.. ")
            break

        # Train
        model.train()
        train_losses = torch.zeros(steps_per_epoch, device=device)
        for i in range(steps_per_epoch):
            try:
                images, _ = next(train_iter)
                images = images.to(device)
            except StopIteration:
                train_iter = iter(train_loader)
                images, _ = next(train_iter)
                images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()
            train_losses[i] = loss

        # Eval
        model.eval()
        val_losses = torch.zeros(val_steps, device=device)
        with torch.no_grad():
            for i in range(val_steps):
                try:
                    images, _ = next(val_iter)
                    images = images.to(device)
                except StopIteration:
                    val_iter = iter(val_loader)
                    images, _ = next(val_iter)
                    images = images.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, images)
                val_losses[i] = loss

        metrics["epoch"].append(epoch + 1)
        metrics["train"].append(train_losses.mean().item())
        metrics["eval"].append(val_losses.mean().item())

        # Save best model
        if metrics["eval"][-1] <= best_val_loss:
            best_val_loss = metrics["eval"][-1]
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': best_val_loss,
            }, best_checkpoint_path)
        else:
            early_stopping_counter += 1

        # Save recent checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': metrics["eval"][-1],
            'early_stopping_counter': early_stopping_counter
        }, latest_checkpoint_path)

        # Save checkpoint every n epochs
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': metrics["eval"][-1],
                'early_stopping_counter': early_stopping_counter
            }, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))

        print("Epoch: {}/{}.. ".format(metrics["epoch"][-1], epochs),
              "Training Loss: {:.3f}.. ".format(metrics["train"][-1]),
              "Validation Loss: {:.3f}.. ".format(metrics["eval"][-1]),
              "Early Stopping Counter: {}/{}.. ".format(early_stopping_counter, early_stopping_patience))

    # Save metrics
    with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    train()
