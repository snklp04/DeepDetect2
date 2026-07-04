"""
ssl_tasks.py — Multimodal SSL Pretraining Orchestrator
======================================================

Orchestrates the Video ViT, Audio Tokenizer, and Fusion Engine.
Applies Cross-Modal Contrastive, AV-Sync, and Masked Modality Learning.
"""

import copy
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContrastiveLearningHead(nn.Module):
    """
    Projects the deeply fused [CLS] tokens into a normalized latent space
    and computes the InfoNCE loss (pulling matching A/V pairs together).
    Uses a memory bank for additional negatives to prevent embedding collapse.
    """
    def __init__(self, embed_dim: int = 768, projection_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim),
        )
        
        # Momentum encoder
        self.momentum = 0.999
        self.projector_m = copy.deepcopy(self.projector)
        for param in self.projector_m.parameters():
            param.requires_grad = False
        
        # Memory bank for additional negatives
        self.queue_size = 2048
        self.register_buffer('queue_v', torch.randn(2048, projection_dim))
        self.register_buffer('queue_a', torch.randn(2048, projection_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize queues after initialization
        self.queue_v = F.normalize(self.queue_v, dim=1)
        self.queue_a = F.normalize(self.queue_a, dim=1)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z_v: torch.Tensor, z_a: torch.Tensor):
        batch_size = z_v.shape[0]
        ptr = int(self.queue_ptr)
        
        # Handle wraparound
        if ptr + batch_size > self.queue_size:
            # Fill remaining space, then wrap
            remaining = self.queue_size - ptr
            self.queue_v[ptr:, :] = z_v[:remaining].detach()
            self.queue_a[ptr:, :] = z_a[:remaining].detach()
            self.queue_v[:batch_size-remaining, :] = z_v[remaining:].detach()
            self.queue_a[:batch_size-remaining, :] = z_a[remaining:].detach()
            ptr = batch_size - remaining
        else:
            self.queue_v[ptr:ptr+batch_size, :] = z_v.detach()
            self.queue_a[ptr:ptr+batch_size, :] = z_a.detach()
            ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum projector with EMA."""
        for param, param_m in zip(self.projector.parameters(), self.projector_m.parameters()):
            param_m.data = self.momentum * param_m.data + (1.0 - self.momentum) * param.data

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projector(features)
        return F.normalize(projected, dim=1)

    @torch.no_grad()
    def forward_momentum(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through momentum projector (no gradients)."""
        projected = self.projector_m(features)
        return F.normalize(projected, dim=1)

    def compute_loss(self, z1: torch.Tensor, z2: torch.Tensor, vis_cls: torch.Tensor, aud_cls: torch.Tensor) -> torch.Tensor:
        # Update momentum encoder
        self._momentum_update()
        
        B = z1.shape[0]
        
        # Compute momentum embeddings for queue (no gradients)
        z1_m = self.forward_momentum(vis_cls)
        z2_m = self.forward_momentum(aud_cls)
        
        # Batch similarities (B x B)
        sim_batch_12 = torch.matmul(z1, z2.T) / self.temperature
        sim_batch_21 = torch.matmul(z2, z1.T) / self.temperature
        
        # Queue similarities (B x queue_size)
        sim_queue_1 = torch.matmul(z1, self.queue_a.T) / self.temperature
        sim_queue_2 = torch.matmul(z2, self.queue_v.T) / self.temperature
        
        # Concatenate: positives on diagonal of batch part, negatives from batch off-diagonal + queue
        logits_12 = torch.cat([sim_batch_12, sim_queue_1], dim=1)
        logits_21 = torch.cat([sim_batch_21, sim_queue_2], dim=1)
        
        # Labels: positive is at index i for sample i
        labels = torch.arange(B, device=z1.device)
        
        loss_12 = F.cross_entropy(logits_12, labels)
        loss_21 = F.cross_entropy(logits_21, labels)
        
        # Update queue with MOMENTUM embeddings (not main embeddings)
        self._dequeue_and_enqueue(z1_m, z2_m)
        
        return (loss_12 + loss_21) / 2.0


class AVSyncPredictor(nn.Module):
    """
    Lightweight Sync Predictor. 
    Because the Fusion Engine already cross-attended the sequences, 
    we just need to classify the concatenated global [CLS] tokens.
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        # Input is concatenated vis_cls and aud_cls (768 * 2 = 1536)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # 0 = Out-of-Sync, 1 = In-Sync
        )

    def forward(self, vis_cls: torch.Tensor, aud_cls: torch.Tensor) -> torch.Tensor:
        fused_context = torch.cat([vis_cls, aud_cls], dim=1)
        return self.classifier(fused_context)


class MaskedModalityHead(nn.Module):
    """
    Reconstructs original tokens from masked outputs.
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.reconstructor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)     
        )

    def forward(self, fused_sequence: torch.Tensor) -> torch.Tensor:
        return self.reconstructor(fused_sequence)


class SSLPretrainModel(nn.Module):
    """
    Complete Mid-Fusion Self-Supervised Pretraining Model.
    """
    def __init__(self, config: dict):
        super().__init__()

        # --- Import your new architecture components here ---
        from Models.Visual_encoder import VideoViTEncoder
        from Models.audio_encoder import AudioTokenEncoder
        from Models.Fusion_Engine import AudioVisualFusionEngine

        self.embed_dim = 768

        # 1. Tokenizing Encoders
        self.visual_encoder = VideoViTEncoder(pretrained=True)
        self.audio_encoder = AudioTokenEncoder(embed_dim=self.embed_dim)

        # 2. The Multimodal Bottleneck Transformer (MBT)
        self.fusion_engine = AudioVisualFusionEngine(
            embed_dim=self.embed_dim, 
            num_layers=4, 
            num_bottlenecks=4
        )

        # 3. Task Heads
        self.contrastive_head = ContrastiveLearningHead(embed_dim=self.embed_dim)
        self.sync_head = AVSyncPredictor(embed_dim=self.embed_dim)
        
        # Give each modality its own dedicated reconstruction head
        self.masked_head_aud = MaskedModalityHead(embed_dim=self.embed_dim)
        self.masked_head_vis = MaskedModalityHead(embed_dim=self.embed_dim) 

        # 4. Masking Parameters (Learnable placeholders for hidden tokens)
        self.mask_token_aud = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.mask_token_vis = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        nn.init.trunc_normal_(self.mask_token_aud, std=0.02)
        nn.init.trunc_normal_(self.mask_token_vis, std=0.02)

    def _generate_block_mask(self, batch_size: int, seq_len: int, mask_ratio: float = 0.7, block_size: int = 4, device=None):
        """
        Generate block masks for masked modeling.
        Instead of random per-token masking, creates contiguous blocks.
        """
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        num_mask = int(seq_len * mask_ratio)
        
        for b in range(batch_size):
            masked_count = 0
            attempts = 0
            while masked_count < num_mask and attempts < 100:
                # Random block start position
                start = torch.randint(0, seq_len - block_size + 1, (1,)).item()
                end = min(start + block_size, seq_len)
                
                # Count how many new tokens this would mask
                new_masks = (~mask[b, start:end]).sum().item()
                if masked_count + new_masks <= num_mask + block_size:  # Allow slight overshoot
                    mask[b, start:end] = True
                    masked_count = mask[b].sum().item()
                attempts += 1
        
        return mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        visual = batch['visual']            # (B, T_v, 3, 224, 224)
        audio = batch['audio_sync']         # (B, 1, n_mels, T_a)
        sync_labels = batch['sync_label']   # (B,)

        losses = {}
        total_loss = torch.tensor(0.0, device=visual.device, requires_grad=True)

        # --- 1. Extract Base Tokens (Un-fused) ---
        v_tokens_original = self.visual_encoder(visual)  # (B, T_v, 768)
        a_tokens_original = self.audio_encoder(audio)    # (B, T_a, 768)

        # --- 2. Dynamic Audio Masking ---
        B, T_a, _ = a_tokens_original.shape
        mask_aud = self._generate_block_mask(B, T_a, mask_ratio=0.80, block_size=12, device=visual.device)
        
        a_tokens_masked = a_tokens_original.clone()
        pos_embeds_aud = self.audio_encoder.temporal_pos_embed[:, :T_a, :].expand(B, -1, -1)
        a_tokens_masked[mask_aud] = self.mask_token_aud.squeeze()

        # --- 3. Dynamic Visual Masking ---
        B, T_v, _ = v_tokens_original.shape
        mask_vis = self._generate_block_mask(B, T_v, mask_ratio=0.85, block_size=8, device=visual.device)
        
        v_tokens_masked = v_tokens_original.clone()
        pos_embeds_vis = self.visual_encoder.temporal_pos_embed[:, :T_v, :].expand(B, -1, -1)
        v_tokens_masked[mask_vis] = self.mask_token_vis.squeeze()

        # --- 4. Deep Fusion (The MBT) ---
        fusion_out = self.fusion_engine(v_tokens_masked, a_tokens_masked)
        
        vis_cls = fusion_out["vis_cls"]
        aud_cls = fusion_out["aud_cls"]
        vis_seq_fused = fusion_out["vis_seq"]
        aud_seq_fused = fusion_out["aud_seq"]

        # --- Task A: Cross-Modal Contrastive Learning ---
        z_visual = self.contrastive_head(vis_cls)
        z_audio = self.contrastive_head(aud_cls)
        
        loss_contrastive = self.contrastive_head.compute_loss(z_visual, z_audio, vis_cls, aud_cls)
        losses['contrastive_loss'] = loss_contrastive
        total_loss = total_loss + (0.5 * loss_contrastive)

        # --- Variance Regularization (Collapse Prevention) ---
        # Penalize if embedding variance drops too low
        std_v = z_visual.std(dim=0).mean()
        std_a = z_audio.std(dim=0).mean()

        # Target std around 0.5, penalize if below 0.1
        var_loss = torch.relu(0.1 - std_v) + torch.relu(0.1 - std_a)
        losses['var_regularization'] = var_loss
        total_loss = total_loss + (0.1 * var_loss)  # Small weight

        # --- Task B: AV Sync Prediction ---
        sync_logits = self.sync_head(vis_cls, aud_cls)
        loss_sync = F.cross_entropy(sync_logits, sync_labels)
        
        losses['av_sync_loss'] = loss_sync
        total_loss = total_loss + (1.0 * loss_sync)

        # --- Task C: Dual Masked Modality Modeling (ALL SAMPLES) ---
        loss_mask_aud = torch.tensor(0.0, device=visual.device, requires_grad=True)
        loss_mask_vis = torch.tensor(0.0, device=visual.device, requires_grad=True)

        # 1. Audio Reconstruction (All samples)
        if mask_aud.any():
            reconstructed_audio = self.masked_head_aud(aud_seq_fused)
            loss_mask_aud = F.smooth_l1_loss(
                reconstructed_audio[mask_aud], 
                a_tokens_original.detach()[mask_aud]
            )

        # 2. Visual Reconstruction (All samples)
        if mask_vis.any():
            reconstructed_visual = self.masked_head_vis(vis_seq_fused)
            loss_mask_vis = F.smooth_l1_loss(
                reconstructed_visual[mask_vis], 
                v_tokens_original.detach()[mask_vis]
            )

        losses['temporal_mask_loss_aud'] = loss_mask_aud
        losses['temporal_mask_loss_vis'] = loss_mask_vis

        total_loss = total_loss + (2.0 * loss_mask_aud) + (3.0 * loss_mask_vis)
        losses['total_loss'] = total_loss
        
        return losses

    def get_encoder_state_dicts(self) -> Dict[str, dict]:
        """
        Extracts the smart encoders AND the fusion engine to pass to Phase 2.
        """
        return {
            'visual': self.visual_encoder.state_dict(),
            'audio': self.audio_encoder.state_dict(),
            'fusion': self.fusion_engine.state_dict(),
        }