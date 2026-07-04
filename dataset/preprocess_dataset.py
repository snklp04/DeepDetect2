import os
import torch
import torchvision.io as io
import torchaudio.transforms as T
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.visual_preprocessing import VisualPreprocessor
from utils.audio_preprocessing import AudioPreprocessor

INPUT_DIR = "voxceleb/real"
OUTPUT_DIR = "voxceleb/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

vproc = VisualPreprocessor(num_frames=16)
aproc = AudioPreprocessor()

for file in tqdm(os.listdir(INPUT_DIR)):
    if not file.endswith(".mp4"):
        continue

    path = os.path.join(INPUT_DIR, file)

    try:
        # 1. Read BOTH video and audio at the same time!
        # pts_unit='sec' is required to avoid the torchvision warning
        video, waveform, info = io.read_video(path, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)

        # 2. Safely get the audio sample rate (default to 16k if missing)
        sr = info.get("audio_fps", 16000) 

        # 3. Process the audio if it exists
        if waveform is not None and waveform.numel() > 0:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = T.Resample(sr, 16000)(waveform)
        else:
            print(f"\nSkipping {file}: No audio track found.")
            continue

        # 4. Run through your preprocessors
        visual = vproc(video.unsqueeze(0))[0]
        audio = aproc(waveform)[0]

        # 5. Fix audio length to exactly 94 steps
        target_steps = 94
        audio = audio[:, :, :target_steps]
        if audio.shape[-1] < target_steps:
            pad = target_steps - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad))

        save_path = os.path.join(OUTPUT_DIR, file.replace(".mp4", ".pt"))
        torch.save({
            "visual": visual,
            "audio": audio
        }, save_path)

    except Exception as e:
        print(f"\nSkipping {file}: {e}")