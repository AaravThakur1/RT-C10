"""
rt_c10.visualization

Visualization utilities for HRMCT experiments.

Contains plotting utilities for

- confusion matrices
- ROC curves
- PCA
- t-SNE
- activation histograms
- training curves
- per-class metrics
- top-k accuracy
- misclassified samples
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

def plot_confusion_matrix(
    matrix,
    class_names,
    figsize=(8, 8),
    cmap="Blues",
):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    matrix : ndarray
        Confusion matrix.

    class_names : list[str]
        Class labels.
    """

    fig, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap=cmap,
    )

    plt.colorbar(
        image,
        ax=ax,
    )

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    return fig, ax


def plot_normalized_confusion_matrix(
    matrix,
    class_names,
    figsize=(8, 8),
):
    """
    Plot a normalized confusion matrix.
    """

    return plot_confusion_matrix(
        matrix,
        class_names,
        figsize=figsize,
        cmap="Blues",
    )


# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------

def plot_pca(
    embeddings,
    labels,
    figsize=(8, 6),
):
    """
    Plot PCA projection of embeddings.

    Returns
    -------
    pca
    transformed_embeddings
    """

    pca = PCA(
        n_components=2,
    )

    reduced = pca.fit_transform(
        embeddings,
    )

    plt.figure(figsize=figsize)

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
    )

    plt.colorbar()

    plt.title("PCA of Embeddings")

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.tight_layout()

    return pca, reduced


# ---------------------------------------------------------
# t-SNE
# ---------------------------------------------------------

def plot_tsne(
    embeddings,
    labels,
    figsize=(8, 6),
    random_state=42,
):
    """
    Plot t-SNE projection of embeddings.
    """

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
    )

    reduced = tsne.fit_transform(
        embeddings,
    )

    plt.figure(figsize=figsize)

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
    )

    plt.colorbar()

    plt.title("t-SNE of Embeddings")

    plt.tight_layout()

    return tsne, reduced

# ---------------------------------------------------------
# ROC Curves
# ---------------------------------------------------------

def plot_roc_curves(
    roc_results,
    figsize=(8, 6),
):
    """
    Plot one-vs-rest ROC curves.

    Parameters
    ----------
    roc_results : dict
        Dictionary returned by
        metrics.compute_roc_auc().
    """

    plt.figure(figsize=figsize)

    for cls in roc_results["fpr"]:

        plt.plot(
            roc_results["fpr"][cls],
            roc_results["tpr"][cls],
            label=f"Class {cls} (AUC = {roc_results['auc'][cls]:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves")

    plt.legend(
        fontsize=8,
        loc="lower right",
    )

    plt.tight_layout()


# ---------------------------------------------------------
# Training Curves
# ---------------------------------------------------------

def plot_loss_curve(
    losses,
    figsize=(8, 5),
):
    """
    Plot training loss.
    """

    plt.figure(figsize=figsize)

    plt.plot(
        losses,
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid(True)

    plt.tight_layout()


def plot_accuracy_curve(
    accuracies,
    figsize=(8, 5),
):
    """
    Plot validation accuracy.
    """

    plt.figure(figsize=figsize)

    plt.plot(
        accuracies,
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Validation Accuracy")

    plt.grid(True)

    plt.tight_layout()


# ---------------------------------------------------------
# Per-class Metrics
# ---------------------------------------------------------

def plot_per_class_accuracy(
    accuracy_dict,
    figsize=(10, 5),
):
    """
    Plot per-class accuracy.
    """

    names = list(
        accuracy_dict.keys()
    )

    values = list(
        accuracy_dict.values()
    )

    plt.figure(figsize=figsize)

    plt.bar(
        names,
        values,
    )

    plt.xticks(rotation=45)

    plt.ylabel("Accuracy")

    plt.title("Per-class Accuracy")

    plt.tight_layout()


def plot_per_class_error(
    error_dict,
    figsize=(10, 5),
):
    """
    Plot per-class error rates.
    """

    names = list(
        error_dict.keys()
    )

    values = list(
        error_dict.values()
    )

    plt.figure(figsize=figsize)

    plt.bar(
        names,
        values,
    )

    plt.xticks(rotation=45)

    plt.ylabel("Error Rate")

    plt.title("Per-class Error")

    plt.tight_layout()


# ---------------------------------------------------------
# Checkpoint Comparison
# ---------------------------------------------------------

def plot_checkpoint_comparison(
    checkpoint_results,
    metric_name="Accuracy",
    figsize=(8, 5),
):
    """
    Compare a metric across checkpoints.

    Parameters
    ----------
    checkpoint_results : dict

        Example

        {
            0: 0.12,
            100: 0.54,
            200: 0.68,
            300: 0.74,
        }
    """

    epochs = list(
        checkpoint_results.keys()
    )

    values = list(
        checkpoint_results.values()
    )

    plt.figure(figsize=figsize)

    plt.plot(
        epochs,
        values,
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Checkpoint Epoch")

    plt.ylabel(metric_name)

    plt.title(
        f"{metric_name} Across Checkpoints"
    )

    plt.grid(True)

    plt.tight_layout()

# ---------------------------------------------------------
# Activation Histograms
# ---------------------------------------------------------

activation_store = {}


def save_activation(name):
    """
    Forward hook used to store layer activations.
    """

    def hook(model, inputs, outputs):

        if hasattr(outputs, "detach"):
            activation_store[name] = (
                outputs.detach()
                .cpu()
                .numpy()
            )

    return hook


def register_activation_hooks(
    model,
):
    """
    Register hooks on every leaf module.

    Returns
    -------
    list
        Hook handles.
    """

    handles = []

    for name, module in model.named_modules():

        if len(list(module.children())) == 0:

            handles.append(
                module.register_forward_hook(
                    save_activation(name)
                )
            )

    return handles


def remove_activation_hooks(
    handles,
):
    """
    Remove previously registered hooks.
    """

    for handle in handles:

        handle.remove()


def plot_activation_histograms(
    bins=50,
    figsize=(8, 4),
):
    """
    Plot histograms of captured activations.
    """

    for name, activation in activation_store.items():

        plt.figure(figsize=figsize)

        plt.hist(
            activation.flatten(),
            bins=bins,
        )

        plt.title(
            f"Activation Histogram\n{name}"
        )

        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")

        plt.tight_layout()

# ---------------------------------------------------------
# Misclassified Samples
# ---------------------------------------------------------

def show_misclassified(
    images,
    predictions,
    labels,
    class_names,
    max_images=9,
    figsize=(10, 10),
):
    """
    Display misclassified images.
    """

    n = min(
        max_images,
        len(images),
    )

    plt.figure(figsize=figsize)

    for i in range(n):

        plt.subplot(
            3,
            3,
            i + 1,
        )

        image = images[i]

        if hasattr(image, "numpy"):
            image = image.numpy()

        image = np.transpose(
            image,
            (1, 2, 0),
        )

        image = (image * 0.5) + 0.5

        plt.imshow(image)

        plt.title(
            f"P: {class_names[predictions[i]]}\n"
            f"T: {class_names[labels[i]]}",
            fontsize=8,
        )

        plt.axis("off")

    plt.tight_layout()

# ---------------------------------------------------------
# Combined Training Curves
# ---------------------------------------------------------

def plot_training_history(
    train_loss,
    validation_accuracy,
    figsize=(10, 4),
):
    """
    Plot loss and validation accuracy together.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    axes[0].plot(
        train_loss,
        linewidth=2,
    )

    axes[0].set_title(
        "Training Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].grid(True)

    axes[1].plot(
        validation_accuracy,
        linewidth=2,
    )

    axes[1].set_title(
        "Validation Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].grid(True)

    plt.tight_layout()

    return fig, axes