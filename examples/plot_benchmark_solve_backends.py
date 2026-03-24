from __future__ import annotations

import argparse
import csv
from math import log2
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "benchmark_solve_backends_n1024.csv"
DEFAULT_OUTPUT = ROOT / "benchmark_solve_backends_n1024.svg"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series(
    rows: list[dict[str, str]],
    mean_key: str,
    stderr_key: str,
) -> tuple[list[int], list[float], list[float]]:
    xs: list[int] = []
    ys_ms: list[float] = []
    errs_ms: list[float] = []
    for row in rows:
        mean = row[mean_key]
        stderr = row[stderr_key]
        if not mean:
            continue
        xs.append(int(row["m"]))
        ys_ms.append(float(mean) * 1000.0)
        errs_ms.append(float(stderr) * 1000.0 if stderr else 0.0)
    return xs, ys_ms, errs_ms


def power_of_two_label(value: int) -> str:
    exponent = round(log2(value))
    if 2**exponent != value:
        raise ValueError(f"Expected a power of two, got {value}.")
    return rf"$2^{{{exponent}}}$"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark timings from examples/benchmark_solve_backends_n1024.csv."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise RuntimeError(f"No benchmark rows found in {args.input}.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.91, bottom=0.17)

    plot_specs = [
        ("mlx-lstsq", "mlx_current_mean_s", "mlx_current_stderr_s", "#0f766e"),
        ("Torch hybrid", "torch_hybrid_mean_s", "torch_hybrid_stderr_s", "#2563eb"),
        ("SciPy CPU", "scipy_cpu_mean_s", "scipy_cpu_stderr_s", "#dc2626"),
        ("NumPy chol", "numpy_chol_mean_s", "numpy_chol_stderr_s", "#7c3aed"),
    ]

    for label, mean_key, stderr_key, color in plot_specs:
        xs, ys_ms, errs_ms = series(rows, mean_key, stderr_key)
        ax.errorbar(
            xs,
            ys_ms,
            yerr=errs_ms,
            label=label,
            color=color,
            marker="o",
            linewidth=2.2,
            markersize=5,
            capsize=2.5,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Least-squares solve time by backend (n=1024)", pad=12)
    ax.set_xlabel("Rows in A (m)")
    ax.set_ylabel("Mean solve time (ms)")

    x_ticks = [int(row["m"]) for row in rows]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([power_of_two_label(x) for x in x_ticks])

    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg", metadata={"Date": None})


if __name__ == "__main__":
    main()
