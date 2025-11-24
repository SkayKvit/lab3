import torch
import torch.nn as nn

# те саме, що в train.py
SAMPLE_RATE = 16000
N_MELS = 64
SAMPLE_CLASSES = [
    "yes", "no", "up", "down",
    "left", "right", "on", "off",
    "stop", "go"
]

class SimpleCNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS, n_classes: int = len(SAMPLE_CLASSES)):
        super(SimpleCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * (n_mels // 4) *  (32 // 4), 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)
        return self.fc(x)
