"""
scripts/compare_checkpoints.py

Compare multiple HRMCT checkpoints on the CIFAR-10 test set.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rt_c10.model import HRMCT
from rt_c10.evaluate import (
    load_checkpoint,
    predict,
)
from rt_c10.metrics import (
    compute_accuracy,
    compute_topk_accuracy,
    compute_precision_recall_f1,
)
from rt_c10.visualization import (
    plot_checkpoint_comparison,
)
from rt_c10.utils import get_device


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "checkpoint_dir",
        help="Directory containing checkpoints.",
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

    checkpoint_dir = Path(args.checkpoint_dir)

    checkpoints = sorted(
        checkpoint_dir.glob("hrmct_ep*.pt")
    )

    if not checkpoints:

        raise FileNotFoundError(
            "No checkpoints found."
        )

    accuracy_history = {}
    top5_history = {}
    f1_history = {}

    for checkpoint in checkpoints:

        epoch = int(
            checkpoint.stem.split("ep")[-1]
        )

        model = HRMCT().to(device)

        load_checkpoint(
            model,
            checkpoint,
            device,
        )

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

        accuracy_history[epoch] = accuracy
        top5_history[epoch] = top5
        f1_history[epoch] = metrics["f1"]

        print(
            f"Epoch {epoch:3d} | "
            f"Acc {accuracy:.4f} | "
            f"Top5 {top5:.4f} | "
            f"F1 {metrics['f1']:.4f}"
        )

    plot_checkpoint_comparison(
        accuracy_history,
        metric_name="Accuracy",
    )

    plot_checkpoint_comparison(
        top5_history,
        metric_name="Top-5 Accuracy",
    )

    plot_checkpoint_comparison(
        f1_history,
        metric_name="F1 Score",
    )

    plt.show()


if __name__ == "__main__":
    main()