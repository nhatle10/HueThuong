"""
Precsise Alignment Loss in Poincare Hyperbolic Space.

Reference:
    PAEFF: Precise Alignment and Enhanced Gated Feature Fusion for Face-Voice Association
    Interspeech 2025 (https://arxiv.org/abs/2505.17002)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import pmath

class ToPoincare(nn.Module):
    """
    Maps Euclidean normalized embeddings into the Pointcare ball
    and scales backprop gradients using the Rimannian conformal factor.
    """
    def __init__(self, c=0.05, riemannian=True):
        super().__init__()
        self.c = c
        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c
        self.grad_fix = (
            lambda x: self.riemannian.apply(x)
        )  if riemannian else (lambda x : x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [1] Project from original space into Poincare ball
        [2] Scale backprop gradients using the Rimannian conformal factor.
        [3] Apply Riemannian gradient correction during backward pass
        """
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))



class PreciseAlignmentLoss(nn.Module):
    """
    Precise Alignment (InfoNCE CLIP-style) loss in Pointcare space
    """
    def __init__(self, curvature: float=0.05, temperature: float=0.07):
        super().__init__()
        self.temperature = temperature
        self.tp = ToPoincare(c=curvature)
    
    def forward(self, face_embeds: torch.Tensor, voice_embeds: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        ----------
            face_embeds: L2-normalized face embeddings [B, D]
            voice_embeds: L2-normalized voice embeddings [B, D]
        Returns:
        --------
            loss: scalar alignment loss
        """
        # [1] Project Euclidean embeddings into Poincare ball
        face_a = self.tp(face_embeds)
        voice_a = self.tp(voice_embeds)

        # [2] Symmetric cross-entropy alignment (CLIP-style InfoNCE)
        logits = (face_a @ voice_a.T) / self.temperature
        labels = torch.arange(face_a.shape[0], device=face_a.device)
        loss_face_to_voice = F.cross_entropy(logits, labels)
        loss_voice_to_face = F.cross_entropy(logits.T, labels)
        
        return (loss_face_to_voice + loss_voice_to_face) / 2.0
