#!/usr/bin/env python3
"""
All credits go to https://github.com/ImprintLab/MedSegDiff/compare/master...MagnusGoltermann:MedSegDiff:master

Plot training progress metrics from a progress.csv file.
Features
- Auto-detects common metric groups (loss, loss_cal, loss_diff, vb) and their quantiles (_q0.._q3)
- Plots scalar metrics (e.g., grad_norm, param_norm, lr) when present
- Optional moving-average smoothing and downsampling for large runs
- Saves figures as PNGs to an output directory next to the CSV by default
Usage examples
- python plot_progress.py                          # uses runs/isic256/progress.csv if it exists
- python plot_progress.py path/to/progress.csv     # plot a specific CSV
- python plot_progress.py -s 21                    # smooth with a 21-step moving average
- python plot_progress.py --max-points 3000        # downsample to ~3000 points for faster plotting
- python plot_progress.py --show                   # also open the figures interactively
Y-scale
- Use --yscale log to draw all grouped plots on a log scale
- The dedicated loss-only plot uses --loss-yscale (default: log)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _import_pandas_and_matplotlib():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pandas is required. Please install with `pip install pandas`."
        ) from exc

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required. Please install with `pip install matplotlib`."
        ) from exc

    return pd, plt


def parse_args() -> argparse.Namespace:
    default_csv = None
    # Heuristic default path commonly used in this project
    candidate = Path("runs") / "isic256" / "progress.csv"
    if candidate.exists():
        default_csv = str(candidate)

    parser = argparse.ArgumentParser(
        description="Plot training metrics from a progress.csv file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        nargs=(None if default_csv is None else "?"),
        default=default_csv,
        help=(
            "Path to progress.csv. If omitted, attempts to use runs/isic256/progress.csv"
            if default_csv is not None
            else "Path to progress.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to <csv_dir>/plots",
    )
    parser.add_argument(
        "-s",
        "--smoothing-window",
        dest="smoothing_window",
        type=int,
        default=0,
        help="Moving average window size. 0 disables smoothing.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=4000,
        help=(
            "If >0, downsample each series to approximately this many points for speed."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Figure DPI when saving PNG files.",
    )
    parser.add_argument(
        "--yscale",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale for grouped plots.",
    )
    parser.add_argument(
        "--loss-yscale",
        type=str,
        choices=["linear", "log"],
        default="log",
        help="Y-axis scale for the dedicated loss-only plot.",
    )
    parser.add_argument(
        "--log-eps",
        type=float,
        default=1e-8,
        help="Floor values to this epsilon when using log scale.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open figures interactively in addition to saving.",
    )

    args = parser.parse_args()

    if args.csv_path is None:
        parser.error("csv_path is required when default file is not present.")

    return args


def ensure_outdir(outdir: Optional[str], csv_path: Path) -> Path:
    if outdir is not None:
        output_dir = Path(outdir)
    else:
        output_dir = csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def detect_step_column(columns: Iterable[str]) -> Optional[str]:
    candidates = [
        "step",
        "global_step",
        "steps",
        "iteration",
        "iter",
        "samples",
        "epoch",
    ]
    for name in candidates:
        if name in columns:
            return name
    return None


def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data
    if window > data.shape[0]:
        window = int(data.shape[0])
        if window <= 1:
            return data
    # Use cumulative sum for efficient moving average
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    # Pad start to keep length
    pad = np.full(window - 1, smoothed[0], dtype=float)
    return np.concatenate([pad, smoothed])


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=int)
    # Choose approximately evenly spaced indices, always include endpoints
    return np.unique(
        np.clip(
            np.round(np.linspace(0, length - 1, num=max_points)).astype(int),
            0,
            length - 1,
        )
    )


def group_quantile_columns(columns: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Return mapping: base_metric -> {"main": col, "q0": col?, ..., "q3": col?}
    For a column name like "loss_cal_q2", base is "loss_cal" and quantile is "q2".
    If a base also exists as a scalar (e.g., "loss_cal"), it is stored under key "main".
    """
    quantile_pattern = re.compile(r"^(?P<base>.+)_q(?P<q>[0-3])$")
    groups: Dict[str, Dict[str, str]] = {}
    for col in columns:
        match = quantile_pattern.match(col)
        if match:
            base = match.group("base")
            q_key = f"q{match.group('q')}"
            groups.setdefault(base, {})[q_key] = col
        else:
            # treat as a possible base metric
            groups.setdefault(col, {})["main"] = col
    return groups


def pick_known_groups(groups: Dict[str, Dict[str, str]]) -> List[Tuple[str, Dict[str, str]]]:
    preferred_order = [
        "loss",
        "loss_cal",
        "loss_diff",
        "vb",
    ]
    ordered: List[Tuple[str, Dict[str, str]]] = []
    for name in preferred_order:
        if name in groups:
            ordered.append((name, groups[name]))
    # Append any other groups that look like metrics (exclude obvious axes)
    for key, spec in groups.items():
        if key in preferred_order:
            continue
        if key in {"step", "steps", "samples", "epoch", "iteration", "iter"}:
            continue
        if key.endswith("_q0") or key.endswith("_q1") or key.endswith("_q2") or key.endswith("_q3"):
            continue
        # Avoid plotting clearly redundant scalar-only helpers like param_norm together here
        ordered.append((key, spec))
    return ordered


def prepare_series(
    values: np.ndarray,
    smoothing_window: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    processed = values.astype(float)
    if smoothing_window and smoothing_window > 1:
        processed = rolling_mean(processed, smoothing_window)
    idx = downsample_indices(processed.shape[0], max_points)
    return idx.astype(float), processed[idx]


def main() -> None:
    args = parse_args()
    pd, plt = _import_pandas_and_matplotlib()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    output_dir = ensure_outdir(args.outdir, csv_path)

    # Read CSV
    df = pd.read_csv(csv_path, low_memory=False)

    # Determine x-axis
    step_col = detect_step_column(df.columns)
    if step_col is None:
        x_all = np.arange(len(df), dtype=float)
        x_label = "index"
    else:
        x_all = df[step_col].to_numpy(dtype=float)
        x_label = step_col

    # Numeric columns only
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

    # Build groups and identify interesting known ones first
    groups_all = group_quantile_columns(numeric_cols)
    groups_ordered = pick_known_groups(groups_all)

    # Style
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        plt.style.use("default")

    # Plot known group metrics (losses, vb, etc.)
    for base_name, spec in groups_ordered:
        # Skip pure axis columns
        if base_name in {"step", "steps", "samples", "epoch", "iteration", "iter"}:
            continue

        # Gather series to plot: main + quantiles if present
        series_to_plot: List[Tuple[str, np.ndarray]] = []
        if "main" in spec:
            series_to_plot.append((base_name, df[spec["main"]].to_numpy(dtype=float)))
        for q in ["q0", "q1", "q2", "q3"]:
            if q in spec:
                series_to_plot.append((f"{base_name}_{q}", df[spec[q]].to_numpy(dtype=float)))

        if not series_to_plot:
            continue

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=args.dpi)
        for label, values in series_to_plot:
            values_to_plot = values
            if args.yscale == "log":
                values_to_plot = np.maximum(values_to_plot, args.log_eps)
            x_idx, y_proc = prepare_series(values_to_plot, args.smoothing_window, args.max_points)
            x_vals = x_all[x_idx.astype(int)] if x_label != "index" else x_idx
            ax.plot(x_vals, y_proc, label=label)

        ax.set_title(f"{base_name} over {x_label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(base_name)
        if args.yscale == "log":
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_file = output_dir / f"{base_name}.png"
        fig.savefig(out_file)
        if args.show:
            plt.show(block=False)
        plt.close(fig)

    # Plot a combined norms figure if present
    norm_names = [name for name in ["grad_norm", "param_norm"] if name in numeric_cols]
    if norm_names:
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=args.dpi)
        for name in norm_names:
            values = df[name].to_numpy(dtype=float)
            x_idx, y_proc = prepare_series(values, args.smoothing_window, args.max_points)
            x_vals = x_all[x_idx.astype(int)] if x_label != "index" else x_idx
            ax.plot(x_vals, y_proc, label=name)
        ax.set_title(f"Norms over {x_label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("value")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_file = output_dir / "norms.png"
        fig.savefig(out_file)
        if args.show:
            plt.show(block=False)
        plt.close(fig)

    # Plot learning rate if present
    lr_candidates = ["lr", "learning_rate"]
    lr_present = [name for name in lr_candidates if name in numeric_cols]
    if lr_present:
        fig, ax = plt.subplots(figsize=(9, 3.5), dpi=args.dpi)
        for name in lr_present:
            values = df[name].to_numpy(dtype=float)
            x_idx, y_proc = prepare_series(values, args.smoothing_window, args.max_points)
            x_vals = x_all[x_idx.astype(int)] if x_label != "index" else x_idx
            ax.plot(x_vals, y_proc, label=name)
        ax.set_title(f"Learning rate over {x_label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("lr")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_file = output_dir / "learning_rate.png"
        fig.savefig(out_file)
        if args.show:
            plt.show(block=False)
        plt.close(fig)

    # Dedicated single-metric plot for main 'loss' only (no quantiles)
    if "loss" in numeric_cols:
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=args.dpi)
        values = df["loss"].to_numpy(dtype=float)
        if args.loss_yscale == "log":
            values = np.maximum(values, args.log_eps)
        x_idx, y_proc = prepare_series(values, args.smoothing_window, args.max_points)
        x_vals = x_all[x_idx.astype(int)] if x_label != "index" else x_idx
        ax.plot(x_vals, y_proc, label="loss", color="#1f77b4")
        ax.set_title(f"loss over {x_label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("loss")
        if args.loss_yscale == "log":
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_file = output_dir / "loss_only.png"
        fig.savefig(out_file)
        if args.show:
            plt.show(block=False)
        plt.close(fig)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()