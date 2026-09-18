"""
main.py
-------
Single entry point for the CLI application.

Usage examples:
    python main.py predict-yield --state Punjab --season Kharif --crop Rice --area 2.5

    python main.py recommend-crop --n 90 --p 42 --k 43 --temperature 20.9 \\
        --humidity 82 --ph 6.5 --rainfall 202.9 --top-n 3
"""

import sys
from src.cli import run

if __name__ == "__main__":
    sys.exit(run())
