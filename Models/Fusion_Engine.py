"""
fusion_engine.py — Multimodal Bottleneck Transformer (MBT)
==========================================================

This module deeply entangles Audio and Visual tokens using a Bottleneck 
architecture. The modalities are not allowed to attend directly to each other;
instead, they both read from and write to a small set of shared "Bottleneck" tokens.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MBTLayer(nn.Module):
    """
    A single layer of the Bottleneck Transformer.
    Processes the visual and audio streams in parallel, allowing them to 
    cross-pollinate ONLY through the shared bottleneck tokens.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # We use standard Transformer Encoder Layers for each stream.
        # batch_first=True is crucial because our inputs are (B, T, D)
        self.vis_attention = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.aud_attention = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self, 
        v_tokens: torch.Tensor, 
        a_tokens: torch.Tensor, 
        bot_tokens: torch.Tensor
    ):
        # 1. Append the shared bottleneck tokens to both streams
        v_in = torch.cat([v_tokens, bot_tokens], dim=1)
        a_in = torch.cat([a_tokens, bot_tokens], dim=1)

        # 2. Self-Attention (Tokens attend to themselves AND the bottlenecks)
        v_out = self.vis_attention(v_in)
        a_out = self.aud_attention(a_in)

        # 3. Split the sequences back apart
        num_bot = bot_tokens.shape[1]
        
        v_res = v_out[:, :-num_bot, :]  # Everything except the last N tokens
        bot_v = v_out[:, -num_bot:, :]  # Only the last N bottleneck tokens

        a_res = a_out[:, :-num_bot, :]
        bot_a = a_out[:, -num_bot:, :]

        # 4. The Fusion Step: Average the updated bottlenecks from both streams
        # This creates the newly synchronized bottleneck for the next layer.
        bot_res = (bot_v + bot_a) / 2.0

        return v_res, a_res, bot_res


class AudioVisualFusionEngine(nn.Module):
    """
    The full MBT Fusion Engine.
    Prepends global [CLS] tokens to the sequences, sets up the bottlenecks,
    and runs them through a stack of MBTLayers.
    """
    def __init__(
        self, 
        embed_dim: int = 768, 
        num_layers: int = 4, 
        num_heads: int = 8, 
        num_bottlenecks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bottlenecks = num_bottlenecks

        # 1. Global Classification Tokens
        # These will act as the final summary vectors for our Contrastive Loss
        self.vis_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.aud_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 2. The Bottleneck Tokens (The "Bulletin Board")
        self.bot_tokens = nn.Parameter(torch.zeros(1, num_bottlenecks, embed_dim))

        # Initialize the parameters
        nn.init.trunc_normal_(self.vis_cls_token, std=0.02)
        nn.init.trunc_normal_(self.aud_cls_token, std=0.02)
        nn.init.trunc_normal_(self.bot_tokens, std=0.02)

        # 3. The Stack of Bottleneck Layers
        self.layers = nn.ModuleList([
            MBTLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        logger.info(f"FusionEngine initialized with {num_bottlenecks} Bottleneck tokens over {num_layers} layers.")

    def forward(self, visual_seq: torch.Tensor, audio_seq: torch.Tensor):
        """
        Args:
            visual_seq: (B, T_v, 768)
            audio_seq:  (B, T_a, 768)
            
        Returns:
            Dict containing the deeply fused sequences and the global CLS tokens.
        """
        B = visual_seq.shape[0]

        # 1. Expand our learnable tokens to match the batch size
        vis_cls = self.vis_cls_token.expand(B, -1, -1)
        aud_cls = self.aud_cls_token.expand(B, -1, -1)
        bots = self.bot_tokens.expand(B, -1, -1)

        # 2. Prepend the CLS tokens to the beginning of the sequences
        v_tokens = torch.cat([vis_cls, visual_seq], dim=1)
        a_tokens = torch.cat([aud_cls, audio_seq], dim=1)

        # 3. Pass through the Bottleneck Transformer Layers
        for layer in self.layers:
            v_tokens, a_tokens, bots = layer(v_tokens, a_tokens, bots)

        # 4. Extract the finalized outputs
        # The 0th index is the CLS token, everything after is the sequence
        final_vis_cls = v_tokens[:, 0, :]
        final_aud_cls = a_tokens[:, 0, :]
        
        final_vis_seq = v_tokens[:, 1:, :]
        final_aud_seq = a_tokens[:, 1:, :]

        return {
            "vis_cls": final_vis_cls,   # (B, 768) -> Use for Contrastive Loss
            "aud_cls": final_aud_cls,   # (B, 768) -> Use for Contrastive Loss
            "vis_seq": final_vis_seq,   # (B, T_v, 768) -> Use for Sync / Masked Modeling
            "aud_seq": final_aud_seq    # (B, T_a, 768) -> Use for Sync / Masked Modeling
        }