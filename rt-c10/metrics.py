"""
rt_c10.metrics

Evaluation utilities for the HRMCT model.

This module provides reusable metric computation and
embedding analysis functions extracted from the original
RT-C10 research notebooks.

Contains:

- checkpoint evaluation
- accuracy
- Top-K accuracy
- precision / recall / F1
- classification reports
- confusion matrices
- per-class metrics
- ROC / AUC
- embedding extraction
- PCA
- t-SNE
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc,
)

from sklearn.preprocessing import label_binarize

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def load_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    device=DEVICE,
):
    """
    Loads a checkpoint into a model.
    """

    checkpoint = Path(checkpoint)

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(state["model_state"])

    model.eval()

    return model

def predict(
    model: nn.Module,
    loader,
    device=DEVICE,
):

    """
    Collect predictions, labels and probabilities.
    """

    model.eval()

    preds = []
    labels = []
    probs = []

    with torch.no_grad():

        for images, target in loader:

            images = images.to(device)
            target = target.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):

                logits = outputs[0]

            else:

                logits = outputs

            probability = torch.softmax(
                logits,
                dim=1,
            )

            prediction = logits.argmax(dim=1)

            preds.append(
                prediction.cpu()
            )

            labels.append(
                target.cpu()
            )

            probs.append(
                probability.cpu()
            )

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    probs = torch.cat(probs).numpy()

    return preds, labels, probs

def evaluate_checkpoint(
    model: nn.Module,
    checkpoint: str,
    loader: DataLoader,
    device=DEVICE,
):

    """
    Complete evaluation of one checkpoint.
    """

    load_checkpoint(
        model,
        checkpoint,
        device,
    )

    predictions, labels, probabilities = predict(
        model,
        loader,
        device,
    )

    return {
        "predictions": predictions,
        "labels": labels,
        "probabilities": probabilities,
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
    }

# ---------------------------------------------------------
# Accuracy Metrics
# ---------------------------------------------------------

def compute_accuracy(
    predictions,
    labels,
):
    """
    Compute Top-1 classification accuracy.
    """

    return accuracy_score(
        labels,
        predictions,
    )


def compute_topk_accuracy(
    probabilities,
    labels,
    k=5,
):
    """
    Compute Top-K accuracy.

    Parameters
    ----------
    probabilities : ndarray
        Softmax probabilities.

    labels : ndarray
        Ground-truth labels.

    k : int
        Number of highest-probability predictions to consider.
    """

    topk = np.argsort(
        probabilities,
        axis=1,
    )[:, -k:]

    correct = 0

    for idx, label in enumerate(labels):

        if label in topk[idx]:

            correct += 1

    return correct / len(labels)


# ---------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------

def compute_precision_recall_f1(
    predictions,
    labels,
    average="macro",
):
    """
    Compute precision, recall and F1 score.
    """

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=average,
        zero_division=0,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_micro_metrics(
    predictions,
    labels,
):

    return compute_precision_recall_f1(
        predictions,
        labels,
        average="micro",
    )


def compute_macro_metrics(
    predictions,
    labels,
):

    return compute_precision_recall_f1(
        predictions,
        labels,
        average="macro",
    )


def compute_weighted_metrics(
    predictions,
    labels,
):

    return compute_precision_recall_f1(
        predictions,
        labels,
        average="weighted",
    )


# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

def compute_classification_report(
    predictions,
    labels,
    class_names,
):
    """
    Return sklearn classification report.

    Returns
    -------
    str
    """

    return classification_report(
        labels,
        predictions,
        target_names=class_names,
        zero_division=0,
    )


def compute_classification_report_dict(
    predictions,
    labels,
    class_names,
):
    """
    Return classification report as dictionary.
    """

    return classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

def compute_confusion_matrix(
    predictions,
    labels,
):
    """
    Compute confusion matrix.
    """

    return confusion_matrix(
        labels,
        predictions,
    )


def compute_normalized_confusion_matrix(
    predictions,
    labels,
):
    """
    Compute row-normalized confusion matrix.
    """

    matrix = confusion_matrix(
        labels,
        predictions,
    ).astype(float)

    matrix /= matrix.sum(
        axis=1,
        keepdims=True,
    )

    return matrix


# ---------------------------------------------------------
# Per-class Metrics
# ---------------------------------------------------------

def compute_per_class_accuracy(
    predictions,
    labels,
    class_names,
):
    """
    Compute accuracy for each class.
    """

    results = {}

    for idx, name in enumerate(class_names):

        mask = labels == idx

        if mask.sum() == 0:

            results[name] = 0.0

            continue

        correct = (
            predictions[mask] == idx
        ).sum()

        results[name] = (
            correct / mask.sum()
        )

    return results


def compute_per_class_error(
    predictions,
    labels,
    class_names,
):
    """
    Compute error rate for every class.
    """

    accuracy = compute_per_class_accuracy(
        predictions,
        labels,
        class_names,
    )

    return {
        cls: 1.0 - acc
        for cls, acc in accuracy.items()
    }

# ---------------------------------------------------------
# ROC / AUC
# ---------------------------------------------------------

def compute_roc_auc(
    probabilities,
    labels,
    num_classes=10,
):
    """
    Compute ROC curve coordinates and AUC score
    for every class.

    Returns
    -------
    dict
        Dictionary containing false positive rates,
        true positive rates and AUC values.
    """

    labels_bin = label_binarize(
        labels,
        classes=np.arange(num_classes),
    )

    fpr = {}
    tpr = {}
    auc_scores = {}

    for i in range(num_classes):

        fpr[i], tpr[i], _ = roc_curve(
            labels_bin[:, i],
            probabilities[:, i],
        )

        auc_scores[i] = auc(
            fpr[i],
            tpr[i],
        )

    macro_auc = roc_auc_score(
        labels_bin,
        probabilities,
        average="macro",
        multi_class="ovr",
    )

    weighted_auc = roc_auc_score(
        labels_bin,
        probabilities,
        average="weighted",
        multi_class="ovr",
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_scores,
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
    }


# ---------------------------------------------------------
# Embedding Extraction
# ---------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device=DEVICE,
):
    """
    Extract latent embeddings from HRMCT.

    Returns
    -------
    embeddings : ndarray
    labels : ndarray
    """

    model.eval()

    embeddings = []
    labels = []

    for images, target in loader:

        images = images.to(device)

        outputs = model(images)

        if not isinstance(outputs, tuple):
            raise RuntimeError(
                "Model must return (logits, embeddings)."
            )

        _, latent = outputs

        embeddings.append(
            latent.cpu().numpy()
        )

        labels.append(
            target.numpy()
        )

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    labels = np.concatenate(
        labels,
        axis=0,
    )

    return embeddings, labels


# ---------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------

def compute_pca(
    embeddings,
    n_components=2,
):
    """
    Perform Principal Component Analysis.

    Returns
    -------
    coordinates
    explained_variance
    """

    pca = PCA(
        n_components=n_components,
    )

    coordinates = pca.fit_transform(
        embeddings,
    )

    return (
        coordinates,
        pca.explained_variance_ratio_,
    )


def compute_tsne(
    embeddings,
    n_components=2,
    random_state=42,
    n_iter=300,
):
    """
    Perform t-SNE projection.

    Returns
    -------
    ndarray
    """

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        n_iter=n_iter,
    )

    return tsne.fit_transform(
        embeddings,
    )