"""
finetune_deepfake.py — Fine-tune SSL Model for Deepfake Detection
=================================================================
Loads pretrained encoders and fine-tunes for binary real/fake classification.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from pathlib import Path
import matplotlib.pyplot as plt  # Added for plotting

# Video/Audio processing
import torchvision
import torchaudio

# Import pretrained components
from Models.Visual_encoder import VideoViTEncoder
from Models.audio_encoder import AudioTokenEncoder
from Models.Fusion_Engine import AudioVisualFusionEngine

from utils.visual_preprocessing import VisualPreprocessor
from utils.audio_preprocessing import AudioPreprocessor


class DeepfakeClassifier(nn.Module):
    """
    Fine-tuning model for deepfake detection.
    Uses pretrained encoders + fusion, adds classification head.
    """
    def __init__(self, pretrained_path: str = "outputs/ssl_encoders_fused.pth", freeze_encoders: bool = False):
        super().__init__()
        
        self.embed_dim = 768
        
        # 1. Load pretrained encoders
        self.visual_encoder = VideoViTEncoder(pretrained=True)
        self.audio_encoder = AudioTokenEncoder(embed_dim=self.embed_dim)
        self.fusion_engine = AudioVisualFusionEngine(
            embed_dim=self.embed_dim,
            num_layers=4,
            num_bottlenecks=4
        )
        
        # Load pretrained weights
        if os.path.exists(pretrained_path):
            print(f"📦 Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'visual' in checkpoint:
                # Saved encoder state dicts format
                self.visual_encoder.load_state_dict(checkpoint['visual'], strict=False)
                self.audio_encoder.load_state_dict(checkpoint['audio'], strict=False)
                self.fusion_engine.load_state_dict(checkpoint['fusion'], strict=False)
            else:
                # Full model state dict - extract encoder parts
                visual_state = {k.replace('visual_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('visual_encoder.')}
                audio_state = {k.replace('audio_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('audio_encoder.')}
                fusion_state = {k.replace('fusion_engine.', ''): v for k, v in checkpoint.items() if k.startswith('fusion_engine.')}
                
                if visual_state:
                    self.visual_encoder.load_state_dict(visual_state, strict=False)
                if audio_state:
                    self.audio_encoder.load_state_dict(audio_state, strict=False)
                if fusion_state:
                    self.fusion_engine.load_state_dict(fusion_state, strict=False)
                    
            print("✅ Pretrained weights loaded successfully!")
        else:
            print(f"⚠️ No pretrained weights found at {pretrained_path}, using random initialization")
        
        # 2. Optionally freeze encoders for linear probing
        if freeze_encoders:
            print("🔒 Freezing encoder weights")
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for param in self.fusion_engine.parameters():
                param.requires_grad = False
        
        # 3. Classification head (takes concatenated CLS tokens)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # Binary output (sigmoid applied in loss)
        )
        
        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: (B, T, 3, 224, 224) - video frames
            audio: (B, 1, n_mels, T_a) - audio spectrogram
        Returns:
            logits: (B, 1) - deepfake probability logits
        """
        # Extract features
        v_tokens = self.visual_encoder(visual)  # (B, T_v, 768)
        a_tokens = self.audio_encoder(audio)    # (B, T_a, 768)
        
        # Fuse modalities
        fusion_out = self.fusion_engine(v_tokens, a_tokens)
        
        vis_cls = fusion_out["vis_cls"]  # (B, 768)
        aud_cls = fusion_out["aud_cls"]  # (B, 768)
        
        # Concatenate and classify
        combined = torch.cat([vis_cls, aud_cls], dim=1)  # (B, 1536)
        logits = self.classifier(combined)  # (B, 1)
        
        return logits


class DeepfakeDataset(Dataset):
    """
    Dataset for real/fake video classification.
    Expects: dataset_fine/real/*.mp4 and dataset_fine/fake/*.mp4
    Uses existing preprocessors from utils/ for consistency with pretraining.
    """
    def __init__(self, root_dir: str, num_frames: int = 16, target_size: int = 224):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.target_size = target_size
        
        # Use the same preprocessors as pretraining
        self.visual_preprocessor = VisualPreprocessor(num_frames=num_frames, image_size=target_size)
        self.audio_preprocessor = AudioPreprocessor(sample_rate=16000, n_mels=128)
        
        # Collect videos with labels
        self.samples = []
        
        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"
        
        if real_dir.exists():
            for video_path in real_dir.glob("*.mp4"):
                self.samples.append((str(video_path), 0))  # 0 = real
        
        if fake_dir.exists():
            for video_path in fake_dir.glob("*.mp4"):
                self.samples.append((str(video_path), 1))  # 1 = fake
        
        random.shuffle(self.samples)
        
        print(f"📂 Found {len(self.samples)} videos")
        print(f"   Real: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"   Fake: {sum(1 for _, l in self.samples if l == 1)}")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No videos found in {root_dir}/real or {root_dir}/fake")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load video frames and preprocess using VisualPreprocessor."""
        try:
            # Read video using torchvision
            video, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')
            # video shape: (T, H, W, C)
            
            T = video.shape[0]
            
            if T == 0:
                raise ValueError("Empty video")
            
            # Convert to (T, C, H, W) format and float
            frames = video.permute(0, 3, 1, 2).float()  # (T, C, H, W)
            
            # Add batch dimension for preprocessor: (1, T, C, H, W)
            frames = frames.unsqueeze(0)
            
            # Use VisualPreprocessor (handles sampling, cropping, normalization)
            processed = self.visual_preprocessor(frames)  # (1, num_frames, 3, 224, 224)
            
            return processed.squeeze(0)  # (num_frames, 3, 224, 224)
            
        except Exception as e:
            print(f"⚠️ Error loading video {video_path}: {e}")
            # Return black frames as fallback
            return torch.zeros(self.num_frames, 3, self.target_size, self.target_size)
    
    def _load_audio(self, video_path: str) -> torch.Tensor:
        """Load audio from video and preprocess using AudioPreprocessor."""
        try:
            # Load audio from video file
            waveform, sample_rate = torchaudio.load(video_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Add batch dimension for preprocessor: (1, num_samples)
            waveform = waveform.squeeze(0).unsqueeze(0)  # (1, num_samples)
            
            # Use AudioPreprocessor (converts to log-mel spectrogram)
            mel_spec = self.audio_preprocessor(waveform)  # (1, 1, n_mels, T)
            
            # Remove batch dimension
            mel_spec = mel_spec.squeeze(0)  # (1, n_mels, T)
            
            # Fix length to 94 frames (same as pretraining)
            target_len = 94
            if mel_spec.shape[-1] > target_len:
                mel_spec = mel_spec[:, :, :target_len]
            elif mel_spec.shape[-1] < target_len:
                pad = target_len - mel_spec.shape[-1]
                mel_spec = F.pad(mel_spec, (0, pad))
            
            return mel_spec  # (1, n_mels, 94)
            
        except Exception as e:
            print(f"⚠️ Error loading audio from {video_path}: {e}")
            # Return silence as fallback
            return torch.zeros(1, 128, 94)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        visual = self._load_video_frames(video_path)
        audio = self._load_audio(video_path)
        
        return {
            "visual": visual,
            "audio": audio,
            "label": torch.tensor(label, dtype=torch.float32)
        }


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Training")
    for batch in loop:
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(visual, audio).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_postfix(loss=loss.item(), acc=correct/total)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        
        logits = model(visual, audio).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        total_loss += loss.item()
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    # Calculate metrics
    accuracy = correct / total
    
    # True positives, false positives, etc.
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_loss / len(loader), accuracy, precision, recall, f1

# --- Added function for plotting ---
def save_metrics_plots(history, total_epochs, save_dir="metrics"):
    """Generates and saves a combined plot of all tracking metrics"""
    epochs_range = range(1, total_epochs + 1)
    plt.style.use('ggplot') # Gives a nice visual style
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Plot
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs_range, history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # 2. Accuracy Plot
    axes[1].plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(epochs_range, history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    # 3. Precision, Recall, F1 Plot
    axes[2].plot(epochs_range, history['precision'], label='Precision', marker='^')
    axes[2].plot(epochs_range, history['recall'], label='Recall', marker='v')
    axes[2].plot(epochs_range, history['f1'], label='F1 Score', marker='D')
    axes[2].set_title('Validation Evaluation Metrics')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics_dashboard.png"), dpi=300)
    plt.close()


def main():
    # Config
    PRETRAINED_PATH = "outputs/ssl_encoders_fused.pth"
    DATASET_DIR = "dataset_fine"
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    FREEZE_ENCODERS = False  # Set True for linear probing first
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    
    # Create metrics folder for the photos
    os.makedirs("metrics", exist_ok=True)
    
    # Initialize history dictionary
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'precision': [], 'recall': [], 'f1': []
    }
    
    # Create model
    model = DeepfakeClassifier(
        pretrained_path=PRETRAINED_PATH,
        freeze_encoders=FREEZE_ENCODERS
    ).to(device)
    
    # Create dataset and split
    full_dataset = DeepfakeDataset(DATASET_DIR)
    
    # 80/20 train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Optimizer with different LR for pretrained vs new layers
    pretrained_params = list(model.visual_encoder.parameters()) + \
                        list(model.audio_encoder.parameters()) + \
                        list(model.fusion_engine.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": LEARNING_RATE * 0.1},  # Lower LR for pretrained
        {"params": classifier_params, "lr": LEARNING_RATE}         # Full LR for classifier
    ], weight_decay=1e-4)
    
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_f1 = 0
    os.makedirs("outputs_finetune", exist_ok=True)
    
    print("\n🚀 Starting Deepfake Detection Fine-tuning...\n")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*50}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc, precision, recall, f1 = evaluate(model, val_loader, device)
        
        # --- Store metrics for this epoch ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        
        scheduler.step()
        
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
                'accuracy': val_acc
            }, "outputs_finetune/best_deepfake_detector.pth")
            print(f"   🏆 New Best Model! F1: {f1:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"outputs_finetune/checkpoint_epoch_{epoch}.pth")
    
    # --- Generate and Save Metric Photos after all Epochs finish ---
    print("\n📸 Generating and saving metrics plots...")
    save_metrics_plots(history, EPOCHS, save_dir="metrics")
    print(f"   Saved dashboard photo to metrics/training_metrics_dashboard.png")

    print(f"\n🎉 Training Complete! Best F1: {best_f1:.4f}")
    print(f"   Model saved to: outputs_finetune/best_deepfake_detector.pth")


if __name__ == "__main__":
    main()