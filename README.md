<div align="center">

# 🎭 DeepDetect2

### Multimodal Audio-Visual Deepfake Detection Framework

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A state-of-the-art self-supervised learning framework for detecting synthetic media through cross-modal audio-visual inconsistency analysis.*

[**Getting Started**](#-quick-start) • [**Architecture**](#-architecture) • [**Results**](#-results) • [**Citation**](#-citation)

</div>

---

## 🌟 Highlights

- 🔥 **Self-Supervised Pretraining** — No labeled data required for representation learning
- 🎯 **Multi-Task Learning** — Contrastive + Sync + Masked Modeling objectives
- 🔗 **Multimodal Bottleneck Transformer** — Efficient cross-modal fusion without expensive attention
- 📈 **State-of-the-Art Performance** — High accuracy on FakeAVCeleb benchmark
- ⚡ **Mixed Precision Training** — Optimized for consumer GPUs (T4/RTX 3060+)

---

## 🎯 The Problem

Modern deepfakes are increasingly convincing visually, but often contain subtle **audio-visual inconsistencies**:

| Artifact Type | Description | Detection Approach |
|---------------|-------------|-------------------|
| **Lip Sync Errors** | Mouth movements don't match speech | Temporal alignment modeling |
| **Audio Splicing** | Real audio + fake video misalignment | Cross-modal contrastive learning |
| **Temporal Glitches** | Unnatural frame transitions | Masked reconstruction |

DeepDetect2 learns to identify these inconsistencies through **self-supervised pretraining**, then fine-tunes for binary real/fake classification.

---

## 🏗️ Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DeepDetect2 Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌──────────────┐              ┌──────────────┐                  │
│    │   Video      │              │    Audio     │                  │
│    │  (16 frames) │              │ (Mel Spec)   │                  │
│    └──────┬───────┘              └──────┬───────┘                  │
│           │                             │                          │
│           ▼                             ▼                          │
│    ┌──────────────┐              ┌──────────────┐                  │
│    │   ViT-B/16   │              │  CNN Encoder │                  │
│    │  (Pretrained)│              │  (4 layers)  │                  │
│    └──────┬───────┘              └──────┬───────┘                  │
│           │                             │                          │
│           │      ┌─────────────────┐    │                          │
│           └─────►│   Multimodal    │◄───┘                          │
│                  │   Bottleneck    │                               │
│                  │   Transformer   │                               │
│                  └────────┬────────┘                               │
│                           │                                        │
│           ┌───────────────┼───────────────┐                        │
│           ▼               ▼               ▼                        │
│    ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│    │ Contrastive│  │  AV Sync   │  │   Masked   │                 │
│    │    Loss    │  │ Prediction │  │  Modeling  │                 │
│    └────────────┘  └────────────┘  └────────────┘                 │
│                                                                     │
│                    SSL PRETRAINING (VoxCeleb)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                  ┌─────────────────────────┐                       │
│                  │   Classification Head   │                       │
│                  │   (Real vs Fake)        │                       │
│                  └─────────────────────────┘                       │
│                                                                     │
│                    FINE-TUNING (FakeAVCeleb)                       │
└─────────────────────────────────────────────────────────────────────┘
```

</div>

### Key Components

| Component | Description | Output Shape |
|-----------|-------------|--------------|
| **Visual Encoder** | ViT-B/16 + Temporal Positional Embeddings | `(B, 16, 768)` |
| **Audio Encoder** | 4-layer CNN → 768-dim projection | `(B, T_a, 768)` |
| **Fusion Engine** | 4-layer MBT with 4 bottleneck tokens | `(B, 768)` per modality |
| **Contrastive Head** | MoCo-style with 2048 memory bank | InfoNCE loss |
| **Classifier** | 3-layer MLP (1536 → 512 → 256 → 1) | Binary prediction |

---

## 📊 Results

### SSL Pretraining Convergence

<div align="center">

| Metric | Initial | Final (15 epochs) |
|--------|---------|-------------------|
| **Total Loss** | 3.6 | 0.55 |
| **Contrastive Loss** | 4.5 | 0.4 |
| **Sync Loss** | 0.65 | 0.05 |
| **Audio Recon** | 0.4 | 0.2 |
| **Visual Recon** | 0.25 | 0.05 |

</div>

### Fine-tuning Performance (FakeAVCeleb)

<div align="center">

| Metric | Score |
|--------|-------|
| **Accuracy** | 85.2% |
| **Precision** | 86.42% |
| **Recall** | 75.6% |
| **F1 Score** | 80.1% |

</div>

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install torch torchvision torchaudio
pip install tqdm matplotlib tensorboard
```

### Installation

```bash
git clone https://github.com/yourusername/DeepDetect2.git
cd DeepDetect2
```

### Training Pipeline

```bash
# Step 1: Download VoxCeleb for pretraining
python dataset/download_pretrain_dataset.py

# Step 2: Preprocess videos
python dataset/preprocess_dataset.py

# Step 3: SSL Pretraining (20K videos, ~50 epochs)
python ssl_pretrain.py

# Step 4: Download FakeAVCeleb for fine-tuning
python dataset/download_finetune_dataset.py

# Step 5: Prepare fine-tuning dataset
python dataset/prepare_finetune_dataset.py

# Step 6: Fine-tune for deepfake detection
python finetune_deepfake.py
```

### Inference

```python
import torch
from Models.Visual_encoder import VideoViTEncoder
from Models.audio_encoder import AudioTokenEncoder
from Models.Fusion_Engine import AudioVisualFusionEngine

# Load trained model
checkpoint = torch.load("outputs_finetune/best_deepfake_detector.pth")
model = DeepfakeClassifier()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    logits = model(video_tensor, audio_tensor)
    probability = torch.sigmoid(logits)
    is_fake = probability > 0.5
```

---

## 📁 Project Structure

```
DeepDetect2/
├── Models/
│   ├── Visual_encoder.py      # ViT-based video encoder
│   ├── audio_encoder.py       # CNN-based audio tokenizer
│   ├── Fusion_Engine.py       # Multimodal Bottleneck Transformer
│   └── ssl_tasks.py           # SSL pretraining orchestrator
│
├── utils/
│   ├── visual_preprocessing.py  # Video frame processing
│   ├── audio_preprocessing.py   # Mel spectrogram extraction
│   └── augmentations.py         # Data augmentation strategies
│
├── dataset/
│   ├── download_pretrain_dataset.py   # VoxCeleb downloader
│   ├── preprocess_dataset.py          # Video → tensor conversion
│   ├── download_finetune_dataset.py   # FakeAVCeleb downloader
│   └── prepare_finetune_dataset.py    # Real/fake organization
│
├── ssl_pretrain.py            # Self-supervised pretraining script
├── finetune_deepfake.py       # Binary classification fine-tuning
│
├── outputs/                   # Pretrained model checkpoints
├── outputs_finetune/          # Fine-tuned model checkpoints
└── metrics/                   # Training visualizations
```

---

## 🔬 Technical Details

### Self-Supervised Learning Objectives

#### 1. Cross-Modal Contrastive Learning
- **Goal**: Align matching audio-video pairs in embedding space
- **Method**: MoCo-style with momentum encoder (m=0.999)
- **Memory Bank**: 2048 negative samples
- **Temperature**: τ = 0.07

#### 2. Audio-Visual Synchronization
- **Goal**: Detect temporal alignment between modalities
- **Negatives**: 30% cross-sample, 70% temporal shift (15-40 frames)
- **Output**: Binary sync/out-of-sync prediction

#### 3. Masked Modality Modeling
- **Visual**: 85% mask ratio, block size 8
- **Audio**: 80% mask ratio, block size 12
- **Loss**: Smooth L1 reconstruction

### Training Configuration

| Parameter | SSL Pretraining | Fine-tuning |
|-----------|-----------------|-------------|
| **Batch Size** | 8 (×4 accumulation) | 4 |
| **Learning Rate** | 1e-4 | 1e-4 (classifier), 1e-5 (encoders) |
| **Optimizer** | AdamW | AdamW |
| **Weight Decay** | 1e-4 | 1e-4 |
| **Epochs** | 50 | 20 |
| **Mixed Precision** | ✅ FP16 | ✅ FP16 |

---

## 🛠️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA T4 (16GB) | RTX 3090/4090 (24GB) |
| **RAM** | 16GB | 32GB |
| **Storage** | 50GB | 100GB (for datasets) |

---

## 📈 Training Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir=runs/ssl_exp

# View metrics
# - Loss curves (total, contrastive, sync, reconstruction)
# - Embedding variance (collapse detection)
# - Learning rate schedule
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 References

- [Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135) — MBT Architecture
- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) — MoCo
- [FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset](https://arxiv.org/abs/2108.05080) — Evaluation Dataset
- [VoxCeleb: Large-scale Speaker Verification in the Wild](https://arxiv.org/abs/1706.08612) — Pretraining Dataset

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out via email.

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for the AI Safety community

</div>
