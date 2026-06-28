# RT-C10: Recursive Transformer for CIFAR-10

RT-C10 is a PyTorch implementation of a **Recursive Transformer** for image classification on the CIFAR-10 dataset. The project investigates recursive latent-space refinement by repeatedly applying a shared transformation to an encoded representation before classification.

Originally developed as an experimental research project, RT-C10 has been refactored into a modular Python package with reusable training, evaluation, visualization, and metrics utilities. The repository includes command-line scripts for training, evaluating, and comparing model checkpoints, together with tools for analyzing learned representations and model behavior throughout training.

---



# Motivation

Transformer-based models have become increasingly successful across computer vision tasks, yet most architectures rely on a fixed sequence of encoder layers before producing a prediction.

RT-C10 explores a simple alternative: after the Transformer encoder produces a latent representation, the representation is recursively refined through a shared transformation before classification. Because the same parameters are reused at every recursive step, additional iterative computation is introduced without increasing the number of learned parameters inside the recursive module.

Beyond implementing the architecture itself, this project emphasizes **model interpretability and analysis**. Rather than reporting only final classification accuracy, the repository includes utilities for evaluating feature representations, embedding structure, checkpoint progression, and multiple complementary performance metrics.

---



# Architecture

The model follows the pipeline below:

```
Input Image
      │
      ▼
Image Flattening
      │
      ▼
Linear Embedding
      │
      ▼
Transformer Encoder
      │
      ▼
Latent Representation
      │
      ▼
Recursive Refinement
(shared transformation applied repeatedly)
      │
      ▼
Classification Head
      │
      ▼
Predicted Class
```

The recursive refinement stage is controlled through a configurable recursion depth. Rather than stacking additional independent layers, the same linear transformation is applied repeatedly to the latent representation before classification.

The implementation intentionally preserves the architecture used during experimentation to maintain reproducibility.

---



# Training

The model was trained on the **CIFAR-10** dataset for **300 epochs** using PyTorch.

The training pipeline includes:

- Automatic checkpoint creation
- Resume-from-checkpoint functionality
- Mixed precision (AMP) training
- Validation after every epoch
- Automatic checkpoint comparison support

Training utilities are separated from executable scripts, allowing the same codebase to be reused for experimentation or future extensions.

---



# Evaluation

RT-C10 includes a modular evaluation pipeline that extends well beyond standard accuracy reporting.

Implemented evaluation methods include:

- Classification Accuracy
- Top-5 Accuracy
- Precision
- Recall
- F1 Score
- Classification Reports
- Confusion Matrices
- Normalized Confusion Matrices
- ROC Curves and AUC
- PCA Embedding Projection
- t-SNE Embedding Visualization
- Per-class Accuracy
- Cross-checkpoint Comparison
- Gradient Norm Analysis
- Activation Distribution Analysis
- Weight Distribution Analysis

These analyses provide insight into convergence behavior, latent-space organization, class-level performance, and model evolution throughout training.

---



# Results

The repository includes visualizations generated throughout training and evaluation, including:

- Training and validation loss curves
- Validation accuracy across checkpoints
- Test accuracy progression
- Per-class accuracy across checkpoints
- Top-5 accuracy
- Precision, Recall, and F1-score comparisons
- Confusion matrices
- ROC curves
- PCA projections of learned embeddings
- t-SNE visualization of latent representations
- Gradient norm evolution
- Representative activation distributions

Example figures are provided in the `figures/` directory.

---



# Pretrained Checkpoints

Because GitHub limits uploaded files to **100 MB**, pretrained model checkpoints are hosted separately.

The released checkpoints correspond to:

- `hrmct_ep000.pt`
- `hrmct_ep100.pt`
- `hrmct_ep200.pt`
- `hrmct_ep300.pt`

Download all checkpoints here:

[https://mega.nz/folder/dJFjUAhR#4IIB4xvbnfQkNVTVIlmX6Q](https://mega.nz/folder/dJFjUAhR#4IIB4xvbnfQkNVTVIlmX6Q)

After downloading, place the checkpoint files inside a local `checkpoints/` directory before running the evaluation or checkpoint comparison scripts.

---



# Installation

```bash
git clone https://github.com/AaravThakur1/RT-C10.git
cd RT-C10

pip install -r requirements.txt
```

---



# Training

Train the model from scratch:

```bash
python scripts/train.py
```

---



# Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py --checkpoint checkpoints/hrmct_ep300.pt
```

---



# Compare Checkpoints

Compare every checkpoint inside a directory:

```bash
python scripts/compare_checkpoints.py checkpoints/
```

The script evaluates each checkpoint individually before plotting performance progression across training.

---



# Future Work

Possible future directions include:

- Patch-based tokenization instead of flattened image embeddings
- Evaluation on larger image classification datasets
- Comparisons against Vision Transformers and CNN baselines
- Investigation of adaptive recursion depth
- Computational efficiency benchmarking
- Ablation studies on recursive refinement depth and shared parameterization

---



# License

Released under the MIT License.