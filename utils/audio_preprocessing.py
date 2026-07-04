"""
audio_preprocessing.py
======================
Converts raw audio waveforms into Log-Mel Spectrograms.
"""

import torch
import torch.nn as nn
import torchaudio


class AudioPreprocessor(nn.Module):
    """
    Transforms raw 16kHz audio waveforms into normalized Log-Mel Spectrograms.
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            n_fft: int = 1024,
            hop_length: int = 512,
            n_mels: int = 128
    ):
        super().__init__()

        # 1. Mel Spectrogram Transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )

        # 2. Convert Power/Amplitude to Decibels (Log scale)
        # Human hearing (and deepfake artifacts) scale logarithmically
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, num_samples) - Raw audio wave (e.g., 3 seconds * 16000 = 48000 samples)

        Returns:
            (B, 1, n_mels, time_steps) - Ready for the AudioEncoder
        """
        # Torchaudio expects (B, channels, time) for waveforms
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, 1, num_samples)

        mel_spec = self.mel_transform(waveform)  # (B, 1, n_mels, T)
        log_mel_spec = self.amplitude_to_db(mel_spec)  # (B, 1, n_mels, T)

        return log_mel_spec