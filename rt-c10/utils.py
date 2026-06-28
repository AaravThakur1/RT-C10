"""
rt_c10.utils

General utility functions used throughout the RT-C10 project.
"""

from __future__ import annotations

from pathlib import Path

import random

import numpy as np
import torch


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

def get_device():
    """
    Return the available PyTorch device.
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------
# Random Seed
# ---------------------------------------------------------

def set_seed(
    seed=42,
):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Checkpoint Directory
# ---------------------------------------------------------

def ensure_directory(
    directory,
):
    """
    Create a directory if it does not exist.
    """

    Path(directory).mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------
# Parameter Count
# ---------------------------------------------------------

def count_parameters(
    model,
):
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ---------------------------------------------------------
# Model Summary
# ---------------------------------------------------------

def print_model_summary(
    model,
):
    """
    Print basic model information.
    """

    print(model)

    print(
        f"\nTrainable Parameters: "
        f"{count_parameters(model):,}"
    )