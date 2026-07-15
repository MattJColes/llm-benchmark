from pathlib import Path

from llm_benchmark.vision_generators import (
    generate_architecture,
    generate_chart,
    generate_floorplan,
    generate_table,
)


def test_generators_are_deterministic(tmp_path: Path) -> None:
    assert generate_chart(tmp_path / "chart.png", seed=42) == generate_chart(
        tmp_path / "chart-2.png", seed=42
    )
    assert generate_table(tmp_path / "table.png", seed=42) == generate_table(
        tmp_path / "table-2.png", seed=42
    )
    assert generate_architecture(tmp_path / "architecture.png", seed=42) == [
        ("client", "api"),
        ("api", "worker"),
        ("worker", "store"),
    ]
    assert generate_floorplan(tmp_path / "floorplan.png", seed=42) == 5
