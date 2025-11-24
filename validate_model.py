import os
import torch
import numpy as np
from scipy.io import wavfile

from train import SimpleCNN, SAMPLE_CLASSES, SAMPLE_RATE, N_MELS

device = torch.device("cpu")

MODEL_PATH = "artifacts/model.pth"
TEST_DIR = os.path.dirname(__file__)


# ------------------------------------------------------------
# Load WAV (без torchaudio, без soundfile)
# ------------------------------------------------------------
def load_audio(path):
    sr, data = wavfile.read(path)

    # Normalize to float32 [-1..1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)

    # Stereo → mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        scale = SAMPLE_RATE / sr
        new_len = int(len(data) * scale)
        data = np.interp(
            np.linspace(0, len(data), new_len),
            np.arange(len(data)),
            data
        )

    wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    return wav


# ------------------------------------------------------------
# Pure Torch Mel-Spectrogram (без torchaudio, без librosa)
# ------------------------------------------------------------
def mel_spectrogram(waveform, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    # STFT
    spec = torch.stft(
        waveform,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        return_complex=True
    ).abs() ** 2  # power

    # Create mel filterbank manually
    # (precomputed a simple triangular mel filterbank)
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    # mel scale bins
    mel_points = np.linspace(
        hz_to_mel(0),
        hz_to_mel(sample_rate // 2),
        n_mels + 2
    )
    hz_points = mel_to_hz(mel_points)

    # FFT bin frequencies
    fft_freqs = np.linspace(0, sample_rate // 2, 1 + 1024 // 2)

    fb = np.zeros((n_mels, len(fft_freqs)))

    for i in range(1, n_mels + 1):
        f_left = hz_points[i - 1]
        f_center = hz_points[i]
        f_right = hz_points[i + 1]

        left_slope = (fft_freqs - f_left) / (f_center - f_left)
        right_slope = (f_right - fft_freqs) / (f_right - f_center)
        fb[i - 1] = np.maximum(0, np.minimum(left_slope, right_slope))

    fb = torch.tensor(fb, dtype=torch.float32)

    mel_spec = fb @ spec.squeeze(0)
    mel_spec = torch.log(mel_spec + 1e-9)

    return mel_spec.unsqueeze(0)  # [1, n_mels, time]


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ------------------------------------------------------------
# Scan wav files near validate_model.py
# ------------------------------------------------------------
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".wav")]

if not test_files:
    print("❌ No .wav files near validate_model.py")
    exit(1)

print(f"Found {len(test_files)} test WAV files\n")

# ------------------------------------------------------------
# Run predictions
# ------------------------------------------------------------
for fname in test_files:
    path = os.path.join(TEST_DIR, fname)

    wav = load_audio(path).to(device)
    mel = mel_spectrogram(wav)
    mel = mel.unsqueeze(0)  # batch

    with torch.no_grad():
        logits = model(mel)
        pred = torch.argmax(logits, dim=1).item()

    print(f"{fname} → {SAMPLE_CLASSES[pred]}")
