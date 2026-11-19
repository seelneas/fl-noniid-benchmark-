
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flbench.data.datasets import download_cifar10


train_dataset, test_dataset = download_cifar10()

print("CIFAR-10 downloaded successfully!")
print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("First image label:", label)
