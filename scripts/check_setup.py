
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.flbench.utils import get_device, set_seed

config_path = PROJECT_ROOT / "configs" / "base.yaml"

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

set_seed(config["seed"])
device = get_device()

print("Configuration loaded successfully!")
print("Project:", config["project"]["name"])
print("Dataset:", config["data"]["dataset"])
print("Number of clients:", config["data"]["num_clients"])
print("Algorithm:", config["federated"]["algorithm"])
print("Selected device:", device)
