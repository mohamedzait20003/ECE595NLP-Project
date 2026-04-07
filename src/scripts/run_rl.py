import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.main.training.rl_train import rl_train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = str(PROJECT_ROOT / "src" / "config" / "rl_config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 RL fine-tuning for CiteMind")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to rl_config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    print(f"Config: {config_path}")
    rl_train(str(config_path))


if __name__ == "__main__":
    main()
