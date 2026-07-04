"""
audio_encoders.py — Transformer-Ready Audio Branch
===================================================

This module implements a CNN-based audio tokenizer. It processes log-Mel 
spectrograms, squashes the frequency domain, and outputs a strict timeline 
of audio tokens. Crucially, it projects these tokens to the same embedding 
dimension as the visual branch and injects Temporal Positional Embeddings.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AudioTokenEncoder(nn.Module):
    """
    Sequence-Aware Audio Tokenizer.

    Compresses the frequency dimension of a Mel-spectrogram while maintaining
    a rich sequence of temporal features. Projects the output into Transformer
    tokens and injects positional awareness.

    Input Shape:  (B, 1, n_mels, time_steps)
    Output Shape: (B, time_steps_out, embed_dim)
    """

    def __init__(
            self,
            input_channels: int = 1,
            embed_dim: int = 768,  # Matches the ViT-Base dimension
            max_audio_frames: int = 256, # Defines the max length for positional embeddings
            dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Convolutional Tokenizer (Compressing Frequency, Preserving Time)
        self.conv_tokenizer = nn.Sequential(
            # Layer 1: Compress time & frequency slightly to reduce sequence length
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2: Compress both again
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3: Rectangular Pooling - Compress frequency ONLY
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Layer 4: Rectangular Pooling - Compress frequency ONLY
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Squash any remaining frequency bins to exactly 1
            nn.AdaptiveAvgPool2d((1, None))
        )

        # 2. Token Projection
        # Projects the 256-dim CNN features up to the 768-dim Transformer space
        self.token_projection = nn.Sequential(
            nn.Linear(256, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout)
        )

        # 3. Temporal Positional Embeddings
        # Crucial for Transformers: teaches the network the chronological order of the audio
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_audio_frames, self.embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        self._init_weights()
        logger.info(
            f"AudioTokenEncoder initialized. "
            f"Output token dimension: (B, T_a, {self.embed_dim})"
        )

    def _init_weights(self):
        for m in self.conv_tokenizer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        for m in self.token_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: (B, 1, n_mels, time_steps)

        Returns:
            (B, T_a, 768) — Chronological sequence of audio tokens
        """
        # 1. Extract Convolutional Features
        x = self.conv_tokenizer(mel_spectrogram)  # (B, 256, 1, T_a)

        # 2. Reshape into a Token Sequence
        x = x.squeeze(2)          # Remove empty frequency dim -> (B, 256, T_a)
        x = x.transpose(1, 2)     # Swap to (B, T_a, 256) for Transformer compatibility

        # 3. Project to Transformer Embedding Dimension (768)
        audio_tokens = self.token_projection(x)  # (B, T_a, 768)

        # 4. Inject Temporal Positional Context
        T_a = audio_tokens.shape[1]
        audio_tokens = audio_tokens + self.temporal_pos_embed[:, :T_a, :]

        return audio_tokens