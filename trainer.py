from __future__ import annotations

import argparse
import subprocess
import sys


COMMANDS = {
    "preprocess": "data.preprocess_localize_mi",
    "mae": "training.train_mae",
    "stad": "training.train_stad",
    "eval": "eval.eval_reconstruction",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-level entry point for the STAD pipeline.")
    parser.add_argument("task", choices=COMMANDS.keys())
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    module = COMMANDS[parsed.task]
    command = [sys.executable, "-m", module, *parsed.args]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
