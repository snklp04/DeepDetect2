"""
visual_preprocessing.py
=======================
Prepares raw video frames for the sequence-aware Visual and Provenance encoders.
Handles temporal sampling, facial cropping, and standard ImageNet normalization.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class VisualPreprocessor(nn.Module):
    """
    Transforms raw video frames into normalized, face-cropped frame sequences
    ready for both the VisualEncoder and ProvenanceEncoder.
    """

    def __init__(
            self,
            num_frames: int = 16,
            image_size: int = 224,
            crop_factor: float = 1.3  # How much context to include around the face
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.crop_factor = crop_factor

        # Standard ImageNet normalization (required since VisualEncoder uses pretrained ResNet)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Resizing transform
        self.resize = transforms.Resize((self.image_size, self.image_size), antialias=True)

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Uniformly samples `num_frames` from the full video sequence to create a consistent timeline.
        Args:
            frames: (T_total, C, H, W)
        Returns:
            (num_frames, C, H, W)
        """
        total_frames = frames.shape[0]
        if total_frames == self.num_frames:
            return frames

        # Generate uniform indices across the timeline
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        return frames[indices]

    def _crop_face_region(self, frames: torch.Tensor, bounding_box: tuple = None) -> torch.Tensor:
        """
        Crops the frames around the face.
        Note: In a production pipeline, bounding_box would be provided by a
        face tracker (like MTCNN or BlazeFace) run prior to this step.
        """
        # If no bounding box is provided, we assume the frames are already loosely
        # centered and perform a center crop as a fallback.
        if bounding_box is None:
            _, _, h, w = frames.shape
            min_dim = min(h, w)
            crop_size = int(min_dim * 0.8)  # 80% center crop

            # Use torchvision's CenterCrop
            center_crop = transforms.CenterCrop(crop_size)
            return center_crop(frames)

        # (Advanced tracking logic would go here to crop using specific coordinates)
        pass

    def forward(self, raw_video: torch.Tensor, face_box: tuple = None) -> torch.Tensor:
        """
        Args:
            raw_video: (B, T_total, 3, H, W) - The raw decoded video batch
            face_box: Optional bounding box coordinates for the face track

        Returns:
            (B, T, 3, 224, 224) - Clean, cropped, normalized sequence
        """
        B, T_total, C, H, W = raw_video.shape
        processed_batch = []

        for b in range(B):
            video_frames = raw_video[b]  # (T_total, 3, H, W)

            # Step 1: Temporal Sampling (Create the strict timeline)
            sampled_frames = self._sample_frames(video_frames)  # (T, 3, H, W)

            # Step 2: Spatial Cropping (Focus only on the face)
            cropped_frames = self._crop_face_region(sampled_frames, face_box)

            # Step 3: Resize to exactly 224x224 for ResNet
            resized_frames = self.resize(cropped_frames)

            # Step 4: Normalize pixel values (0-1 range to ImageNet stats)
            if resized_frames.max() > 1.0:
                resized_frames = resized_frames / 255.0

            normalized_frames = torch.stack([self.normalize(f) for f in resized_frames])

            processed_batch.append(normalized_frames)

        # Re-stack into batch format
        final_tensor = torch.stack(processed_batch, dim=0)

        return final_tensor  # (B, T, 3, 224, 224)