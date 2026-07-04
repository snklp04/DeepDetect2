"""
visual_encoders.py — Transformer-Based Visual Branch
====================================================

This module extracts spatial tokens from video frames using a pre-trained
Vision Transformer (ViT). It then injects Temporal Positional Embeddings 
so the sequence maintains strict chronological order for downstream fusion.
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class VideoViTEncoder(nn.Module):
    """
    Sequence-Aware Vision Transformer for Video.

    Processes frames individually through a ViT to extract the [CLS] token 
    (a global summary of the frame), then applies temporal positional embeddings
    across the time dimension.

    Input Shape:  (B, T, 3, 224, 224)
    Output Shape: (B, T, embed_dim)
    """

    def __init__(
        self,
        num_frames: int = 16,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1. Load Pre-trained ViT-Base (Patch size 16x16)
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        
        # The embedding dimension of ViT-Base is 768
        self.embed_dim = self.vit.hidden_dim
        
        # We don't need the final classification head for ImageNet classes (1000 dims)
        # We replace it with an Identity layer so we just get the raw 768-dim [CLS] token
        self.vit.heads = nn.Sequential(nn.Identity())
        
        # 2. Temporal Positional Embeddings
        # The ViT has spatial embeddings, but we need to teach it time.
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, self.embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"VideoViTEncoder initialized. "
            f"Output sequence shape: (B, {num_frames}, {self.embed_dim})"
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, 224, 224)

        Returns:
            (B, T, 768) — Chronological sequence of frame tokens
        """
        assert frames.dim() == 5, f"Expected 5D input (B, T, C, H, W), got {frames.dim()}D"
        
        B, T, C, H, W = frames.shape

        # 1. Flatten time into the batch dimension to process all frames
        x = frames.reshape(B * T, C, H, W)  # (B*T, 3, 224, 224)

        # 2. Extract spatial [CLS] tokens using the ViT
        # ViT outputs the classification token when passed through our Identity head
        x = self.vit(x)  # (B*T, 768)

        # 3. Reshape back into a strict temporal sequence
        timeline_tokens = x.reshape(B, T, self.embed_dim)  # (B, T, 768)

        # 4. Inject Temporal Context
        # Add the learnable temporal positional embeddings
        timeline_tokens = timeline_tokens + self.temporal_pos_embed[:, :T, :]
        
        timeline_tokens = self.dropout(timeline_tokens)

        return timeline_tokens