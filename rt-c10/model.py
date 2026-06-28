"""
model.py

Model definition for the Hybrid Recursive Modeling Classification Transformer (HRMCT) used in the RT-C10 project.
"""

import torch
import torch.nn as nn


class HRMCT(nn.Module):
    """
    Hybrid Recursive Modeling Classification Transformer (HRMCT).

    A Transformer encoder followed by recursive latent refinement
    for CIFAR-10 image classification.
    """

    def __init__(
        self,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        num_classes=10,
        recursion_depth=2,
    ):
        super().__init__()

        self.embedding = nn.Linear(32 * 32 * 3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.token_head = nn.Linear(d_model, num_classes)
        self.self_head = nn.Linear(d_model, d_model)

        self.recursion_depth = recursion_depth

    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, -1)

        h = self.embedding(x).unsqueeze(1)

        h = self.transformer(h)

        h_pred = h

        for _ in range(self.recursion_depth):
            h_pred = self.self_head(h_pred)

        token_logits = self.token_head(h_pred.squeeze(1))

        return token_logits, h_pred.squeeze(1)