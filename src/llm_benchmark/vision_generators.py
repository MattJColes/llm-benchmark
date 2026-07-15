"""Deterministic synthetic VLM asset generators."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot


def generate_chart(path: Path, *, seed: int) -> float:
    randomizer = random.Random(seed)
    value = round(randomizer.uniform(10, 90), 1)
    figure, axis = pyplot.subplots(figsize=(6, 4), dpi=100)
    axis.bar(["2023", "2024"], [round(value * 0.8, 1), value])
    axis.set_ylabel("Value")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    pyplot.close(figure)
    return value


def generate_table(path: Path, *, seed: int) -> list[list[int]]:
    randomizer = random.Random(seed)
    values = [[randomizer.randrange(1, 100) for _ in range(3)] for _ in range(3)]
    figure, axis = pyplot.subplots(figsize=(4, 2), dpi=100)
    axis.axis("off")
    axis.table(cellText=values, loc="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)
    return values


def generate_architecture(path: Path, *, seed: int) -> list[tuple[str, str]]:
    nodes = ["client", "api", "worker", "store"]
    edges = [(nodes[index], nodes[index + 1]) for index in range(3)]
    figure, axis = pyplot.subplots(figsize=(6, 2), dpi=100)
    axis.axis("off")
    for index, node in enumerate(nodes):
        axis.text(index, 0.5, node, ha="center", bbox={"boxstyle": "round", "facecolor": "white"})
    for index in range(3):
        axis.annotate(
            "", xy=(index + 0.8, 0.5), xytext=(index + 0.2, 0.5), arrowprops={"arrowstyle": "->"}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)
    return edges


def generate_floorplan(path: Path, *, seed: int) -> int:
    randomizer = random.Random(seed)
    room_count = randomizer.randrange(3, 6)
    figure, axis = pyplot.subplots(figsize=(6, 4), dpi=100)
    axis.set_aspect("equal")
    for index in range(room_count):
        axis.add_patch(pyplot.Rectangle((index * 2, 0), 2, 2, fill=False))
        axis.text(index * 2 + 1, 1, f"Room {index + 1}", ha="center", va="center")
    axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)
    return room_count
