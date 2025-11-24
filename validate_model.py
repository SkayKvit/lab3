# validate_model.py
import time
import json
import torch
import torchaudio
import os
from model import SimpleCNN, SAMPLE_CLASSES, SAMPLE_RATE, N_MELS  # ті ж самі конфіги

MODEL_PATH = "artifacts/model.pth"
TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # директорія з тестовими wav-файлами

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# MelSpectrogram трансформація (як в app.py)
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)

def process_file(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    spec = mel_transform(waveform).squeeze(0)
    if spec.dim() == 2:
        spec = spec.unsqueeze(0)  # [1, n_mels, time]
    return spec.to(device)

# збір усіх wav-файлів з TEST_DIR
test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(".wav")]

latencies = []
predictions = []

for file_path in test_files:
    x = process_file(file_path)
    # вимірюємо latency 5 разів для стабільності
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy().tolist()[0]
            top_idx = int(torch.argmax(outputs))
        end = time.time()
        latencies.append(end - start)
    predictions.append({
        "file": os.path.basename(file_path),
        "predicted": SAMPLE_CLASSES[top_idx],
        "probabilities": dict(zip(SAMPLE_CLASSES, probs))
    })

metrics = {
    "avg_latency_sec": sum(latencies)/len(latencies),
    "min_latency_sec": min(latencies),
    "max_latency_sec": max(latencies),
    "num_samples": len(test_files),
    "predictions": predictions
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to metrics.json")
print(metrics)
