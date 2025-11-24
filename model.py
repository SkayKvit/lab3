# model.py
import torch
import torch.nn as nn

SAMPLE_RATE = 16000
N_MELS = 64
SAMPLE_CLASSES = [
    "yes", "no", "up", "down",
    "left", "right", "on", "off",
    "stop", "go"
]

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=len(SAMPLE_CLASSES)):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        # !!! РОЗМІРИ ЯК У ОРИГІНАЛЬНІЙ МОДЕЛІ
        self.fc1 = nn.Linear(16 * 16 * 16, 128)   # перевіримо нижче
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
