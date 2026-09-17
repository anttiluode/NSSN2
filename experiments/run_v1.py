from __future__ import annotations

import argparse
import json
from pathlib import Path

from nssn2.experiment_v1 import run_v1_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NSSN2 v1 prediction-residual gate.")
    parser.add_argument("--episodes", type=int, default=360)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("results/v1.json"))
    args = parser.parse_args()

    receipt = run_v1_suite(seeds=tuple(range(args.seeds)), episodes=args.episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
