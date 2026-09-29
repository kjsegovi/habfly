"""Build a standalone offline mascot preview; never start an agent or browser."""

import argparse
from pathlib import Path

from habfly.browser_mascot import preview_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/mascot-preview/index.html"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(preview_html())
    except FileExistsError:
        parser.error("Preview already exists; open it or choose a new --output path")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
