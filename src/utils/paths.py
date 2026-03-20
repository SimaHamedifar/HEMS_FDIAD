from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

def get_dir(key):
    path = PROJECT_ROOT / config["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path