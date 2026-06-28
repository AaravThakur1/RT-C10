"""
scripts/evaluate.py

Evaluate a trained HRMCT checkpoint on the CIFAR-10 test set.
"""

import argparse

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rt_c10.model import HRMCT
from rt_c10.metrics import (
    compute_accuracy,
    compute_topk_accuracy,
    compute_precision_recall_f1,
    compute_classification_report,
    compute_confusion_matrix,
    compute_normalized_confusion_matrix,
)
from rt_c10.evaluate import (
    load_checkpoint,
    predict,
)
from rt_c10.visualization import (
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
)
from rt_c10.utils import get_device


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint (.pt)",
    )

    args = parser.parse_args()

    device = get_device()

    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    model = HRMCT().to(device)

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    print(f"Loaded checkpoint: {args.checkpoint}")

    logits, predictions, labels = predict(
        model,
        test_loader,
        device,
    )

    probabilities = (
        logits.softmax(dim=1)
        .cpu()
        .numpy()
    )

    accuracy = compute_accuracy(
        predictions,
        labels,
    )

    top5 = compute_topk_accuracy(
        probabilities,
        labels,
        k=5,
    )

    metrics = compute_precision_recall_f1(
        predictions,
        labels,
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Top-5    : {top5:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    print(
        "\nClassification Report\n"
    )

    print(
        compute_classification_report(
            predictions,
            labels,
            test_dataset.classes,
        )
    )

    cm = compute_confusion_matrix(
        predictions,
        labels,
    )

    plot_confusion_matrix(
        cm,
        test_dataset.classes,
    )

    cm_norm = compute_normalized_confusion_matrix(
        predictions,
        labels,
    )

    plot_normalized_confusion_matrix(
        cm_norm,
        test_dataset.classes,
    )

    plt.show()


if __name__ == "__main__":
    main()