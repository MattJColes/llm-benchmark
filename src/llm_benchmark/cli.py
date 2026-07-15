"""Command-line entry point for the benchmark harness."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.parse_args()
