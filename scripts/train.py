"""
scripts/train.py

Train the Hybrid Recursive Modeling Classification Transformer (HRMCT)
on the CIFAR-10 dataset.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_EPOCHS = 301
CHECKPOINT_DIR = "checkpoints"
KEY_EPOCHS = [200, 300]

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rt_c10.model import HRMCT
from rt_c10.train import (
    resume_from_latest,
    save_checkpoint,
    should_save_checkpoint,
    train_one_epoch,
    validate,
)


def main():

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    validation_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
    )

    print(
        f"Train samples: {len(train_dataset)} | "
        f"Validation samples: {len(validation_dataset)}"
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = HRMCT().to(device)

    print(model)

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()

    # -------------------------------------------------
    # Checkpoints
    # -------------------------------------------------

    checkpoint_dir = "checkpoints"

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    start_epoch = resume_from_latest(
        model,
        optimizer,
        checkpoint_dir,
        device,
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    NUM_EPOCHS = 301

    KEY_EPOCHS = [200, 300]

    if start_epoch in KEY_EPOCHS:
        save_checkpoint(
            model,
            optimizer,
            start_epoch,
            checkpoint_dir,
        )

    for epoch in range(start_epoch, NUM_EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        validation_accuracy = validate(
            model,
            validation_loader,
            device,
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS - 1} | "
            f"Loss: {train_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

        if should_save_checkpoint(
            epoch,
            KEY_EPOCHS,
        ):

            save_checkpoint(
                model,
                optimizer,
                epoch,
                checkpoint_dir,
            )


if __name__ == "__main__":
    main()