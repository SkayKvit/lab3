# validate_model.py
import time, json
import torch
import torchaudio
from model import MyModel

def get_sample_input(file_path):
    waveform, sr = torchaudio.load(file_path)
    mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40)(waveform)
    return mfcc.unsqueeze(0)

model = MyModel()
model.load_state_dict(torch.load("artifacts/model.pth", map_location="cpu"))
model.eval()

test_files = ["tests/yes_1.wav", "tests/down_1.wav"]
latencies = []

for f in test_files:
    x = get_sample_input(f)
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        end = time.time()
        latencies.append(end - start)

metrics = {
    "avg_latency_sec": sum(latencies)/len(latencies),
    "min_latency_sec": min(latencies),
    "max_latency_sec": max(latencies),
    "num_samples": len(test_files)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics:", metrics)
