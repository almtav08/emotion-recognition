"""
Definición de Funciones de Pérdida
Focal Loss Multiclase adaptada a la detección de expresiones atípicas e hipomimia en TEA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        """
        Pérdida Focal: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        Obliga a la red a concentrarse en ejemplos difíciles (expresiones tenues)
        y evita saturación por clases fáciles.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()
