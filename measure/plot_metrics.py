from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot runtime measurements from CSV.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("measure/results/runtime_chart.png"))
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    elapsed = [float(row["elapsed_s"]) for row in rows]
    series = [
        ("Processing rate (FPS)", [float(row["fps"]) for row in rows]),
        ("CPU temperature (C)", [float(row["cpu_temperature_c"]) for row in rows]),
        ("Memory use (%)", [float(row["memory_percent"]) for row in rows]),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.4), sharex=True)
    for axis, (label, values) in zip(axes, series):
        axis.plot(elapsed, values, color="black", linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(True, color="0.85", linewidth=0.6)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle("Raspberry Pi Runtime Measurements")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())