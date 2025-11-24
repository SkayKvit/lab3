import os
import torch
import soundfile as sf
import torch.nn.functional as F
from train import SimpleCNN, SAMPLE_CLASSES, SAMPLE_RATE, N_MELS

device = torch.device("cpu")

MODEL_PATH = "artifacts/model.pth"
TEST_DIR = os.path.dirname(__file__)   # шукаємо файли поруч

print("Test dir:", TEST_DIR)


# -------------------------
# Load audio WITHOUT torchaudio
# -------------------------
def load_audio(path):
    data, sr = sf.read(path)

    # stereo → mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # ресемплінг при потребі
    if sr != SAMPLE_RATE:
        import numpy as np
        import math

        scale = SAMPLE_RATE / sr
        new_len = int(len(data) * scale)
        data = np.interp(
            np.linspace(0, len(data), new_len),
            np.arange(len(data)),
            data
        )
        sr = SAMPLE_RATE

    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    return waveform


# -------------------------
# Torch MelSpectrogram (без torchaudio)
# -------------------------
class MelSpecTorch(torch.nn.Module):
    def __init__(self, sample_rate, n_mels):
        super().__init__()
        self.mel = torch.nn.functional.melspectrogram

    def forward(self, wav):
        spec = torch.stft(
            wav,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            return_complex=True,
        )
        spec = spec.abs() ** 2

        # Create mel filterbank manually
        mel_fb = torch.tensor(
            librosa.filters.mel(
                sr=SAMPLE_RATE,
                n_fft=1024,
                n_mels=N_MELS
            ),
            dtype=torch.float32
        )

        mel_spec = torch.matmul(mel_fb, spec.squeeze(0))
        mel_spec = torch.log(mel_spec + 1e-9)
        return mel_spec.unsqueeze(0)


# -------------------------
# Load model
# -------------------------

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

mel_extractor = MelSpecTorch(SAMPLE_RATE, N_MELS)


# -------------------------
# Run prediction
# -------------------------

test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".wav")]

if len(test_files) == 0:
    print("❌ No .wav files found near validate_model.py")
    exit(1)

print(f"Found {len(test_files)} test wav files\n")

for fname in test_files:
    path = os.path.join(TEST_DIR, fname)

    wav = load_audio(path).to(device)
    mel = mel_extractor(wav)
    mel = mel.unsqueeze(0)  # batch

    with torch.no_grad():
        logits = model(mel)
        pred = torch.argmax(logits, dim=1).item()

    print(f"{fname} → {SAMPLE_CLASSES[pred]}")
