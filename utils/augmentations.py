"""
augmentations.py — Data Augmentation for SSL Pretraining
========================================================
Provides video and audio augmentations to prevent trivial solutions.
"""

import torch
import torch.nn.functional as F
import random


class VideoAugmentor:
    """
    Augments video frames to create diverse views.
    Input: (T, 3, H, W) tensor
    """
    def __init__(self, p_flip=0.5, p_blur=0.1):
        self.p_flip = p_flip
        self.p_blur = p_blur
        
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # Random horizontal flip (all frames consistently)
        if random.random() < self.p_flip:
            video = torch.flip(video, dims=[-1])  # Flip width dimension
        
        # Color jitter (applied per-frame but with same params)
        if random.random() < 0.8:
            video = self._color_jitter(video)
        
        # Gaussian blur (rare)
        if random.random() < self.p_blur:
            video = self._gaussian_blur(video)
            
        return video
    
    def _color_jitter(self, video: torch.Tensor) -> torch.Tensor:
        # Brightness
        brightness = random.uniform(0.6, 1.4)
        video = video * brightness
        
        # Contrast
        contrast = random.uniform(0.6, 1.4)
        mean = video.mean(dim=[-1, -2], keepdim=True)
        video = (video - mean) * contrast + mean
        
        # Saturation (for RGB)
        saturation = random.uniform(0.6, 1.4)
        gray = video.mean(dim=1, keepdim=True)  # Grayscale
        video = (video - gray) * saturation + gray
        
        return torch.clamp(video, 0, 1)
    
    def _gaussian_blur(self, video: torch.Tensor) -> torch.Tensor:
        # Simple 3x3 blur kernel
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                              dtype=video.dtype, device=video.device) / 16.0
        kernel = kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3)
        
        T, C, H, W = video.shape
        video = video.view(T * C, 1, H, W)
        video = F.pad(video, (1, 1, 1, 1), mode='reflect')
        video = F.conv2d(video, kernel[:1], groups=1)
        video = video.view(T, C, H, W)
        return video


class AudioAugmentor:
    """
    Augments audio spectrograms (SpecAugment-style).
    Input: (1, n_mels, T) tensor
    """
    def __init__(self, num_time_masks=2, num_freq_masks=2, time_mask_param=10, freq_mask_param=15):
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        # Random gain adjustment (±3dB)
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.4)
            audio = audio * gain
        
        # SpecAugment: Time masking
        for _ in range(self.num_time_masks):
            audio = self._time_mask(audio)
        
        # SpecAugment: Frequency masking
        for _ in range(self.num_freq_masks):
            audio = self._freq_mask(audio)
            
        return audio
    
    def _time_mask(self, audio: torch.Tensor) -> torch.Tensor:
        _, n_mels, T = audio.shape
        t = random.randint(0, min(self.time_mask_param, T - 1))
        t0 = random.randint(0, T - t)
        audio[:, :, t0:t0+t] = 0
        return audio
    
    def _freq_mask(self, audio: torch.Tensor) -> torch.Tensor:
        _, n_mels, T = audio.shape
        f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
        f0 = random.randint(0, n_mels - f)
        audio[:, f0:f0+f, :] = 0
        return audio
