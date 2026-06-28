"""
rt_c10/evaluate.py

Reusable evaluation utilities for HRMCT.

This module intentionally contains NO plotting code.
It only runs inference and returns predictions, probabilities,
embeddings and losses so that metrics.py and visualization.py
can reuse the same evaluation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------

def load_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Load a checkpoint into a model.

    Returns
    -------
    epoch : int
        Saved epoch number.
    """

    checkpoint = Path(checkpoint)

    ckpt = torch.load(checkpoint, map_location=device)

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])

    model.eval()

    return ckpt.get("epoch", 0)


# ---------------------------------------------------------
# Generic inference
# ---------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """
    Run inference.

    Returns
    -------
    logits
    predictions
    labels
    """

    model.eval()

    logits_all = []
    preds_all = []
    labels_all = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        preds = logits.argmax(dim=1)

        logits_all.append(logits.cpu())
        preds_all.append(preds.cpu())
        labels_all.append(labels.cpu())

    return (
        torch.cat(logits_all),
        torch.cat(preds_all),
        torch.cat(labels_all),
    )


# ---------------------------------------------------------
# Embedding inference
# ---------------------------------------------------------

@torch.no_grad()
def predict_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """
    Returns logits, embeddings, predictions and labels.

    Assumes model returns

        logits, embeddings
    """

    model.eval()

    logits_all = []
    embeds_all = []
    preds_all = []
    labels_all = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if not isinstance(outputs, tuple):
            raise RuntimeError(
                "Model must return (logits, embeddings)."
            )

        logits, embeddings = outputs

        preds = logits.argmax(dim=1)

        logits_all.append(logits.cpu())
        embeds_all.append(embeddings.cpu())
        preds_all.append(preds.cpu())
        labels_all.append(labels.cpu())

    return (
        torch.cat(logits_all),
        torch.cat(embeds_all),
        torch.cat(preds_all),
        torch.cat(labels_all),
    )


# ---------------------------------------------------------
# Probabilities
# ---------------------------------------------------------

@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """
    Returns

        probabilities
        labels

    Used for ROC/AUC.
    """

    logits, _, labels = predict(
        model,
        loader,
        device,
    )

    probs = torch.softmax(logits, dim=1)

    return probs, labels


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):

    _, preds, labels = predict(
        model,
        loader,
        device,
    )

    return (preds == labels).float().mean().item()


# ---------------------------------------------------------
# Loss
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Average loss over a dataset.
    """

    model.eval()

    total_loss = 0.0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        loss = criterion(logits, labels)

        batch = labels.size(0)

        total_loss += loss.item() * batch
        total += batch

    return total_loss / total


# ---------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------

@torch.no_grad()
def per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
):
    """
    Returns

        {
            class_name: accuracy
        }
    """

    _, preds, labels = predict(
        model,
        loader,
        device,
    )

    results = {}

    for idx, name in enumerate(class_names):

        mask = labels == idx

        if mask.sum() == 0:
            results[name] = 0.0
            continue

        correct = (preds[mask] == idx).sum().item()

        results[name] = correct / mask.sum().item()

    return results


# ---------------------------------------------------------
# Misclassified samples
# ---------------------------------------------------------

@torch.no_grad()
def misclassified_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """
    Returns

        indices
        predictions
        labels

    Useful for visualization.py.
    """

    model.eval()

    indices = []
    preds_all = []
    labels_all = []

    running_index = 0

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        preds = logits.argmax(dim=1).cpu()

        labels_cpu = labels.cpu()

        incorrect = preds != labels_cpu

        batch_indices = torch.arange(
            running_index,
            running_index + len(labels),
        )

        indices.extend(batch_indices[incorrect].tolist())
        preds_all.extend(preds[incorrect].tolist())
        labels_all.extend(labels_cpu[incorrect].tolist())

        running_index += len(labels)

    return indices, preds_all, labels_all