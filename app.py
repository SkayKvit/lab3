# app.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import torchaudio
import io
import numpy as np

from train import SimpleCNN, SAMPLE_CLASSES, SAMPLE_RATE, N_MELS  # імпорт моделей/конфігів

MODEL_PATH = "/app/artifacts/model.pth"

app = FastAPI(title="Speech commands API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        waveform, sr = torchaudio.load(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"cannot read audio file: {e}"}
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    with torch.no_grad():
        spec = mel_transform(waveform).squeeze(0)  # [n_mels, time]
        # pad/truncate to fixed size (model expects variable but batch needed)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)  # [1, n_mels, time]
        inputs = spec.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().tolist()[0]
        top_idx = int(np.argmax(probs))
        return {"predicted": SAMPLE_CLASSES[top_idx], "probabilities": dict(zip(SAMPLE_CLASSES, probs))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
