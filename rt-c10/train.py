"""
train.py

Training utilities for the RT-C10 project.

This module contains reusable functions for checkpoint management,
training, and validation. Dataset creation and experiment configuration
are handled by scripts/train.py.
"""

import glob
import os

import torch
from torch.cuda.amp import autocast


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """
    Save a model checkpoint.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"hrmct_ep{epoch:03d}.pt",
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint: {checkpoint_path}")


def resume_from_latest(model, optimizer, checkpoint_dir, device):
    """
    Resume training from the latest available checkpoint.

    Returns
    -------
    int
        Epoch number to begin training from.
    """

    checkpoints = sorted(
        glob.glob(
            os.path.join(
                checkpoint_dir,
                "hrmct_ep*.pt",
            )
        )
    )

    if not checkpoints:
        print("No checkpoints found. Starting fresh.")
        return 0

    latest_checkpoint = checkpoints[-1]

    checkpoint = torch.load(
        latest_checkpoint,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["opt_state"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"Resumed from {latest_checkpoint}")
    print(f"Starting at epoch {start_epoch}")

    return start_epoch


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    device,
):
    """
    Train the model for one epoch.

    Returns
    -------
    float
        Average training loss.
    """

    model.train()

    running_loss = 0.0

    for data, target in train_loader:

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        with autocast():

            token_logits, _ = model(data)

            loss = criterion(
                token_logits,
                target,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(
    model,
    validation_loader,
    device,
):
    """
    Evaluate classification accuracy on the validation set.

    Returns
    -------
    float
        Validation accuracy.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for data, target in validation_loader:

            data = data.to(device)
            target = target.to(device)

            token_logits, _ = model(data)

            _, prediction = torch.max(
                token_logits,
                dim=1,
            )

            correct += (prediction == target).sum().item()
            total += target.size(0)

    model.train()

    return correct / total


def should_save_checkpoint(epoch, key_epochs=None, interval=5):
    """
    Determine whether a checkpoint should be saved.

    Parameters
    ----------
    epoch : int
        Current training epoch.
    key_epochs : list[int], optional
        Epochs that should always be saved.
    interval : int
        Save interval in epochs.

    Returns
    -------
    bool
    """

    if key_epochs is None:
        key_epochs = [200, 300]

    return (epoch % interval == 0) or (epoch in key_epochs)