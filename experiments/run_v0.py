from __future__ import annotations

import argparse
import json
from pathlib import Path

from nssn2.experiment import run_v0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NSSN2 v0 developmental-structure gate.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--episodes", type=int, default=360)
    parser.add_argument("--out", type=Path, default=Path("results/v0.json"))
    args = parser.parse_args()

    receipt = run_v0(seed=args.seed, episodes=args.episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
