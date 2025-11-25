import argparse
import os
import time
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# ---------- конфіг ----------
SAMPLE_CLASSES = ["yes", "no", "up", "down"]
DATA_PATH = "/data/speech_commands"   # у контейнері
SAMPLE_RATE = 16000
N_MELS = 32
BATCH_SIZE = 8
EPOCHS = 1
ARTIFACT_DIR = "/artifacts"


# ---------- функції ----------

def filter_valid_files(file_list, base_path):
    valid_files = []
    for filepath in file_list:
        full_path = os.path.join(base_path, filepath)
        try:
            waveform, sr = torchaudio.load(full_path)
            valid_files.append(filepath)
        except Exception as e:
            print(f"Skipped file {full_path} due to error: {e}")
    return valid_files


class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, subset: str = None, download=False):
        super().__init__(DATA_PATH, download=download)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return [line.strip() for line in f]

        if subset == "validation":
            walker = load_list("validation_list.txt")
        elif subset == "testing":
            walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            walker = [w for w in self._walker if w not in excludes]
        else:
            walker = self._walker

        self._walker = filter_valid_files(walker, self._path)
        print(f"{subset} set: {len(self._walker)} valid files")

    def __getitem__(self, n):
        filepath = os.path.join(self._path, self._walker[n])
        label = os.path.basename(os.path.dirname(filepath))

        try:
            waveform, sample_rate = torchaudio.load(filepath)
        except Exception as e:
            print(f"Skipped file {filepath} due to error: {e}")
            return None

        return waveform, sample_rate, label, filepath


def label_to_index(word):
    return torch.tensor(SAMPLE_CLASSES.index(word))


def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[2] in SAMPLE_CLASSES]
    if not batch:
        return torch.empty(0), torch.empty(0)

    max_len = max([waveform.size(1) for waveform, _, _, *_ in batch])
    tensors, targets = [], []

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)

    for waveform, _, label, *_ in batch:
        try:
            spec = transform(waveform).squeeze(0)
        except Exception as e:
            print(f"Skipped file due to error: {e}")
            continue

        if spec.size(1) < max_len:
            spec = nn.functional.pad(spec, (0, max_len - spec.size(1)))

        tensors.append(spec)
        targets.append(label_to_index(label))

    if tensors:
        return torch.stack(tensors), torch.tensor(targets)
    else:
        return torch.empty(0), torch.empty(0)


# ---------- модель ----------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(SAMPLE_CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------- main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-model", help="Path to save model", default=None)
    parser.add_argument("--download-data", action="store_true", help="Download dataset via torchaudio")
    args = parser.parse_args()

    # створити директорію тільки під час тренування
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # датасети
    train_set = SubsetSC("training", download=args.download_data)
    test_set = SubsetSC("testing", download=False)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )

    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---- тренування ----
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            if inputs.size(0) == 0:
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss / max(1, len(train_loader)):.4f}")

    # ---- тест ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs.size(0) == 0:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        accuracy = 100 * correct / total
        print(f"Accuracy = {accuracy:.2f}%")

    # ---- зберегти ----
    if args.save_model:
        save_path = args.save_model
    else:
        save_path = os.path.join(ARTIFACT_DIR, "model.pth")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")
