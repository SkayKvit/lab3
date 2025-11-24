# model.py
import torch
import torch.nn as nn

SAMPLE_RATE = 16000
N_MELS = 64

# !!! У ТВОЇЙ МОДЕЛІ Є ТІЛЬКИ 4 КЛАСИ !!!
SAMPLE_CLASSES = ["yes", "no", "up", "down"]

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=len(SAMPLE_CLASSES)):
        super(SimpleCNN, self).__init__()

        # Відповідає параметрам у model.pth
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        # 64 → 32 → 16 (мел-частота)
        # 32 → 16 → 8  (часова довжина)
        # тому 32 * 16 * 8 = 4096?  → але у твоїй моделі 2048
        #
        # значить ти подавав 32 x 32 spectrogram!
        # 32 → 16 → 8 → 4
        # 32 * 4 * 16 = 2048 ← точно збігається

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
