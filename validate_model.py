import os
import wave
import time
import json
import numpy as np
import torch

from train import SimpleCNN, SAMPLE_CLASSES, SAMPLE_RATE, N_MELS

device = torch.device("cpu")

MODEL_PATH = "artifacts/model.pth"
TEST_DIR = os.path.dirname(__file__)
OUTPUT_JSON = "validation_results.json"


# ------------------------------------------------------------
# WAV loader using Python standard library only
# ------------------------------------------------------------
def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    audio /= 32768.0

    if sr != SAMPLE_RATE:
        scale = SAMPLE_RATE / sr
        new_len = int(len(audio) * scale)
        audio = np.interp(
            np.linspace(0, len(audio), new_len),
            np.arange(len(audio)),
            audio
        )

    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


# ------------------------------------------------------------
# Pure Torch Mel Spectrogram (no torchaudio)
# ------------------------------------------------------------
def mel_spectrogram(waveform, n_mels=N_MELS, sr=SAMPLE_RATE):
    spec = torch.stft(
        waveform,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        return_complex=True,
    ).abs() ** 2

    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(m): return 700 * (10**(m / 2595) - 1)

    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr // 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fft_freqs = np.linspace(0, sr // 2, 1024 // 2 + 1)
    fb = np.zeros((n_mels, len(fft_freqs)))

    for i in range(1, n_mels + 1):
        left, center, right = hz_points[i - 1:i + 2]
        left_slope = (fft_freqs - left) / (center - left)
        right_slope = (right - fft_freqs) / (right - center)
        fb[i - 1] = np.maximum(0, np.minimum(left_slope, right_slope))

    fb = torch.tensor(fb, dtype=torch.float32)
    mel = fb @ spec.squeeze(0)
    mel = torch.log(mel + 1e-9)
    return mel.unsqueeze(0)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ------------------------------------------------------------
# Find WAV test files
# ------------------------------------------------------------
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".wav")]

if not test_files:
    print("❌ No WAV files found")
    exit(1)

print(f"Found {len(test_files)} test files\n")

results = []


# ------------------------------------------------------------
# Perform inference on each WAV file
# ------------------------------------------------------------
for fname in test_files:
    path = os.path.join(TEST_DIR, fname)

    wav = load_wav(path).to(device)
    mel = mel_spectrogram(wav)
    mel = mel.unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        logits = model(mel)
    latency = time.time() - start

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    pred_idx = int(np.argmax(probs))
    pred_class = SAMPLE_CLASSES[pred_idx]

    print(f"{fname} → {pred_class}  ({latency:.4f}s)")

    results.append({
        "file": fname,
        "predicted": pred_class,
        "probabilities": dict(zip(SAMPLE_CLASSES, probs)),
        "logits": logits.cpu().numpy().tolist()[0],
        "latency_sec": latency
    })


# ------------------------------------------------------------
# Save JSON log
# ------------------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUTPUT_JSON)
