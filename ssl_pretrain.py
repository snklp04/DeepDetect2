import os
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Import the newly upgraded MBT model
from Models.ssl_tasks import SSLPretrainModel
from utils.augmentations import VideoAugmentor, AudioAugmentor

# =========================
# CONFIGURATION (Upgraded for ViT + MBT)
# =========================
CONFIG = {
    'model': {
        # Feature dims synced to 768 for seamless Transformer integration
        'visual': {'feature_dim': 768, 'pretrained': True, 'dropout': 0.1},
        'audio': {'feature_dim': 768, 'input_channels': 1, 'dropout': 0.1},
    },
    'ssl': {
        'tasks': {
            'av_sync': {'enabled': True, 'weight': 1.0},
            'contrastive': {'enabled': True, 'weight': 0.5},
            # Replaced temporal_order with Masked Modality Modeling
            'temporal_order': {'enabled': False}, 
        }
    }
}

# =========================
# DATASET (Streamlined)
# =========================
class SSLDataset(Dataset):
    """
    Streamlined Dataset. 
    No longer needs to shuffle frames manually because the new Masked Modality 
    task masks tokens dynamically on the GPU during the forward pass.
    """
    def __init__(self, root_dir):
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ]

        if len(self.files) == 0:
            raise RuntimeError("No processed .pt files found in dataset directory!")
        
        # Initialize augmentors
        self.video_aug = VideoAugmentor()
        self.audio_aug = AudioAugmentor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        visual = data["visual"]
        audio = data["audio"]

        # Apply augmentations BEFORE sync logic
        visual = self.video_aug(visual)
        audio = self.audio_aug(audio)

        # --- AV Sync Augmentation ---
        if random.random() < 0.5:
            # OUT-OF-SYNC (label=0)
            if random.random() < 0.3:
                # Cross-sample negative: use different video's audio
                other_idx = random.randint(0, len(self.files) - 1)
                while other_idx == idx:
                    other_idx = random.randint(0, len(self.files) - 1)
                other_data = torch.load(self.files[other_idx])
                audio_sync = other_data["audio"]
            else:
                # Truncation shift (not roll!) - removes start, pads end with zeros
                shift = random.randint(15, 40)
                audio_sync = torch.nn.functional.pad(
                    audio[:, :, shift:],  # Remove first 'shift' frames
                    (0, shift)            # Pad end with zeros
                )
            sync_label = torch.tensor(0, dtype=torch.long)
        else:
            # IN-SYNC (label=1)
            audio_sync = audio
            sync_label = torch.tensor(1, dtype=torch.long)

        return {
            "visual": visual,
            "audio_sync": audio_sync,
            "sync_label": sync_label
        }

# =========================
# TRAINING SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# NOTE: Adjust root_dir to match your actual dataset path
dataset = SSLDataset("dataset/processed")

loader = DataLoader(
    dataset,
    batch_size=8,                 
    shuffle=True,
    num_workers=8,                
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

model = SSLPretrainModel(CONFIG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = GradScaler()

# =========================
# LOGGING & TRACKING
# =========================
os.makedirs("outputs", exist_ok=True)
writer = SummaryWriter("runs/ssl_exp")

history = {
    "total": [],
    "sync": [],
    "mask_aud": [],      # Track Audio Mask
    "mask_vis": [],      # Track Visual Mask
    "contrast": [],
    "var_reg": []        # Track Variance Regularization
}

def save_plot():
    plt.figure(figsize=(10, 6))
    plt.plot(history["total"], label="Total Loss", linewidth=2)
    plt.plot(history["sync"], label="Sync Loss", linestyle='--')
    plt.plot(history["mask_aud"], label="Audio Recon Loss", linestyle='-.', alpha=0.7)
    plt.plot(history["mask_vis"], label="Visual Recon Loss", linestyle='-.', alpha=0.7)
    plt.plot(history["contrast"], label="Contrastive Loss", linestyle='--')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Multimodal SSL Training Progress")
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/loss_curve.png")
    plt.close()

# =========================
# TRAIN EPOCH
# =========================
def train_epoch(epoch, loader, model, optimizer, scaler, accumulation_steps=4):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    epoch_losses = {"total": 0.0, "sync": 0.0, "mask_aud": 0.0, "mask_vis": 0.0, "contrast": 0.0, "var_reg": 0.0}
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(loop):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with autocast():
            losses = model(batch)
            total_loss = losses["total_loss"] / accumulation_steps  # Scale loss
        
        scaler.scale(total_loss).backward()
        
        # Only step optimizer every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Track unscaled losses for logging
        epoch_losses["total"] += losses["total_loss"].item()
        epoch_losses["sync"] += losses.get("av_sync_loss", torch.tensor(0.0)).item()
        epoch_losses["mask_aud"] += losses.get("temporal_mask_loss_aud", torch.tensor(0.0)).item()
        epoch_losses["mask_vis"] += losses.get("temporal_mask_loss_vis", torch.tensor(0.0)).item()
        epoch_losses["contrast"] += losses.get("contrastive_loss", torch.tensor(0.0)).item()
        epoch_losses["var_reg"] += losses.get("var_regularization", torch.tensor(0.0)).item()
        
        loop.set_postfix(loss=losses["total_loss"].item())
    
    # Handle remaining gradients if not divisible
    if len(loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    n = len(loader)
    for k in epoch_losses.keys():
        epoch_losses[k] /= n
    
    return epoch_losses

# =========================
# MAIN TRAINING LOOP
# =========================
epochs = 50
best_loss = float("inf")
prev_losses = None

print("🚀 Starting Multimodal Bottleneck Pretraining...\n")

for epoch in range(1, epochs + 1):
    avg_losses = train_epoch(epoch, loader, model, optimizer, scaler, accumulation_steps=4)

    # Update history
    history["total"].append(avg_losses["total"])
    history["sync"].append(avg_losses["sync"])
    history["mask_aud"].append(avg_losses["mask_aud"])
    history["mask_vis"].append(avg_losses["mask_vis"])
    history["contrast"].append(avg_losses["contrast"])
    history["var_reg"].append(avg_losses["var_reg"])

    # TensorBoard
    writer.add_scalar("Loss/Total", avg_losses["total"], epoch)
    writer.add_scalar("Loss/Sync", avg_losses["sync"], epoch)
    writer.add_scalar("Loss/Masked_Recon_Aud", avg_losses["mask_aud"], epoch)
    writer.add_scalar("Loss/Masked_Recon_Vis", avg_losses["mask_vis"], epoch)
    writer.add_scalar("Loss/Contrastive", avg_losses["contrast"], epoch)
    writer.add_scalar("Loss/Var_Regularization", avg_losses["var_reg"], epoch)

    # --- DIAGNOSTIC: Check if learning ---
    print(f"\n📊 Epoch {epoch} Summary:")
    print(f"   Total Loss:      {avg_losses['total']:.4f}")
    
    tasks_to_track = ["sync", "mask_aud", "mask_vis", "contrast", "var_reg"]
    
    if prev_losses:
        for task in tasks_to_track:
            diff = prev_losses[task] - avg_losses[task]
            trend = "📉 Dropped" if diff > 0 else "📈 Rose"
            status = "✅ Learning" if diff > 0.005 else ("⚠️ Stagnant" if abs(diff) < 0.005 else "❌ Diverging")
            print(f"   - {task.capitalize()} Loss: {avg_losses[task]:.4f} ({trend} by {abs(diff):.4f}) -> {status}")
    else:
        for task in tasks_to_track:
             print(f"   - {task.capitalize()} Loss: {avg_losses[task]:.4f} (Baseline established)")

    prev_losses = avg_losses

    # Checkpointing
    torch.save(model.state_dict(), f"outputs/model_epoch_{epoch}.pth")

    if avg_losses["total"] < best_loss:
        best_loss = avg_losses["total"]
        torch.save(model.state_dict(), "outputs/best_model.pth")
        print("   🏆 New Best Model Saved!")

    save_plot()
    print("-" * 50)

# Save the finalized encoders + fusion engine for the downstream task
torch.save(model.get_encoder_state_dicts(), "outputs/ssl_encoders_fused.pth")

print("\n🎉 Training Complete! Check 'outputs/loss_curve.png' to visualize the convergence.")