import argparse
import json
import torch
import torch.nn as nn

# === Конфіг (має збігатися з train.py) ===
SAMPLE_CLASSES = ["yes", "no", "up", "down"]
N_MELS = 32
TEST_WIDTH = 160   # умовна довжина мел-спектрограми


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(SAMPLE_CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,mel,time)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def run_inference(model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 1. Створюємо випадковий вхід, як мел-спектрограму ===
    dummy_input = torch.randn(1, N_MELS, TEST_WIDTH).to(device)

    # === 2. Проганяємо модель ===
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1)
        predicted_class = probs.argmax(dim=1).item()

    # === 3. Готуємо метрики ===
    metrics = {
        "predicted_class": SAMPLE_CLASSES[predicted_class],
        "confidences": {cls: float(probs[0][i]) for i, cls in enumerate(SAMPLE_CLASSES)}
    }

    # === 4. Зберігаємо у JSON (як очікує GitHub Actions) ===
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✔ Inference test completed. Metrics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.pth")
    parser.add_argument("--output", required=True, help="Where to save metrics.json")
    args = parser.parse_args()

    run_inference(args.model, args.output)
