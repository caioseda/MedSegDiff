from pathlib import Path
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Reuse visualization helpers from overlay_results for consistent overlays
from overlay_results import (
    get_img_gt_and_pred_paths_from_video_frame_id,
    load_mask_binary,
    visualize_case,
)


def binarize_mask(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load a GT mask as binary ndarray and return array plus (W, H) size."""
    mask_img = Image.open(path).convert("L")
    mask_arr = (np.asarray(mask_img) >= 128).astype(np.uint8)
    if mask_arr.mean() > 0.5:
        mask_arr = 1 - mask_arr
    return mask_arr, mask_img.size  # size is (W, H)


def resolve_preds_dir(solver: str, preds_dir_arg: str) -> Path:
    """Resolve prediction directory, handling nested run folders."""
    default_pred_map = {
        "dpm": "/Users/caioseda/Documents/Trabalho/Tecgraf/projetos/MedSegDiff/data/out_sample_fast",
        "nodpm": "/Users/caioseda/Documents/Trabalho/Tecgraf/projetos/MedSegDiff/data/out_sample/sample_100000step_1000diffsteps_no_dpm_solver",
    }
    base = Path(preds_dir_arg) if preds_dir_arg else Path(default_pred_map[solver])
    if base.is_dir():
        direct_matches = list(base.glob("*_output_ens.*"))
        if direct_matches:
            return base
        # Search one level deep for a run folder with predictions
        for sub in base.iterdir():
            if sub.is_dir() and list(sub.glob("*_output_ens.*")):
                return sub
    return base


def dice_and_iou(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)

    intersection = np.logical_and(gt_bool, pred_bool).sum()
    gt_sum = gt_bool.sum()
    pred_sum = pred_bool.sum()
    union = np.logical_or(gt_bool, pred_bool).sum()

    dice = (2.0 * intersection + eps) / (gt_sum + pred_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def select_extremes(records: List[Dict], metric_key: str) -> Tuple[Dict[str, Dict], float]:
    values = np.array([r[metric_key] for r in records], dtype=float)
    mean_value = float(values.mean())

    best_idx = int(values.argmax())
    worst_idx = int(values.argmin())
    mean_idx = int(np.abs(values - mean_value).argmin())

    selected = {
        "best": records[best_idx],
        "mean": records[mean_idx],
        "worst": records[worst_idx],
    }
    return selected, mean_value


def plot_examples(
    cases: Dict[str, Dict],
    metric_key: str,
    mean_value: float,
    output_dir: Path,
    alpha: float,
    vis_mode: str,
    solver_tag: str,
) -> Dict[str, Path]:
    os.makedirs(output_dir, exist_ok=True)
    metric_name = metric_key.upper()
    saved_paths: Dict[str, Path] = {}

    for label, rec in cases.items():
        metric_value = rec[metric_key]
        suptitle = f"[{solver_tag}] {label.title()} {metric_name}={metric_value:.3f} (id: {rec['id']})"
        out_path = output_dir / f"{rec['id']}_{label}_{metric_key}_{solver_tag}.png"
        visualize_case(
            rec["image_path"],
            rec["mask_path"],
            rec["pred_path"],
            str(out_path),
            alpha=alpha,
            mode=vis_mode,
            suptitle=suptitle,
        )
        print(f"Saved {label} example to {out_path}")
        saved_paths[label] = out_path

    return saved_paths


def plot_triplet_grid(saved_paths: Dict[str, Path], metric_key: str, mean_value: float, output_dir: Path, solver_tag: str) -> None:
    """Combine best/mean/worst example images into a single vertical 3-panel figure."""
    metric_name = metric_key.upper()
    fig = plt.figure(figsize=(7, 12), dpi=150)
    labels = ["best", "mean", "worst"]
    for idx, label in enumerate(labels):
        ax = plt.subplot(3, 1, idx + 1)
        img = np.asarray(Image.open(saved_paths[label]))
        ax.imshow(img)
        # Place label to the left of the panel instead of on top
        ax.text(-0.02, 0.5, label.title(), va="center", ha="right", fontsize=12, transform=ax.transAxes)
        ax.axis("off")

    fig.suptitle(f"[{solver_tag}] {metric_name} best/mean/worst (mean={mean_value:.3f})")
    plt.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.02, hspace=0.02)
    out_path = output_dir / f"{metric_key}_triplet_{solver_tag}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved triplet figure to {out_path}")


def compute(args) -> None:
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    preds_dir = resolve_preds_dir(args.solver, args.preds_dir)
    output_dir = Path(args.output_dir) / args.solver
    os.makedirs(output_dir, exist_ok=True)

    video_frame_ids = sorted(Path(file).stem for file in os.listdir(images_dir))

    records: List[Dict] = []
    skipped = 0

    for idx, video_frame_id in enumerate(video_frame_ids):
        if args.limit and idx >= args.limit:
            break

        image_path, mask_path, pred_path = get_img_gt_and_pred_paths_from_video_frame_id(
            video_frame_id=video_frame_id,
            img_dir=images_dir,
            gt_mask_dir=masks_dir,
            pred_mask_dir=preds_dir,
        )

        if not os.path.exists(mask_path) or not os.path.exists(pred_path):
            skipped += 1
            continue

        try:
            gt_mask, mask_size = binarize_mask(mask_path)
            pred_mask = load_mask_binary(pred_path, size=mask_size)
            dice, iou = dice_and_iou(gt_mask, pred_mask)
            records.append(
                {
                    "id": video_frame_id,
                    "dice": dice,
                    "iou": iou,
                    "miou": iou,  # alias for backward compatibility
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "pred_path": pred_path,
                }
            )
        except Exception as e:
            print(f"Failed on {video_frame_id}: {e}")
            skipped += 1

    if not records:
        print("No samples processed.")
        return

    dice_values = np.array([r["dice"] for r in records], dtype=float)
    iou_values = np.array([r["iou"] for r in records], dtype=float)

    print(f"Processed: {len(records)}, Skipped: {skipped}")
    summary_lines = [
        f"Processed: {len(records)}, Skipped: {skipped}",
        f"Dice    -> mean: {dice_values.mean():.4f}, std: {dice_values.std():.4f}, min: {dice_values.min():.4f}, max: {dice_values.max():.4f}",
        f"mIoU    -> mean: {iou_values.mean():.4f}, std: {iou_values.std():.4f}, min: {iou_values.min():.4f}, max: {iou_values.max():.4f}",
    ]
    for line in summary_lines[1:]:
        print(line)

    summary_path = output_dir / f"metrics_summary_dice_iou_{args.solver}.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved summary to {summary_path}")

    if args.plot_extremes:
        metric_key = "iou" if args.plot_metric == "miou" else args.plot_metric
        cases, mean_value = select_extremes(records, metric_key)
        examples_dir = output_dir / f"examples_{metric_key}_{args.solver}"
        saved_paths = plot_examples(
            cases,
            metric_key,
            mean_value,
            examples_dir,
            alpha=args.alpha,
            vis_mode=args.vis_mode,
            solver_tag=args.solver,
        )
        if set(saved_paths.keys()) == {"best", "mean", "worst"}:
            plot_triplet_grid(saved_paths, metric_key, mean_value, output_dir, args.solver)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Dice and mIoU for predicted masks vs ground truth.")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/Users/caioseda/Documents/Trabalho/Tecgraf/projetos/MedSegDiff/vfss-data-split/data/dataset_inca/test/data/",
        help="Directory containing test images (e.g., <video_frame_id>*.png)",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="/Users/caioseda/Documents/Trabalho/Tecgraf/projetos/MedSegDiff/vfss-data-split/data/dataset_inca/test/target/",
        help="Directory containing GT masks (e.g., <video_frame_id>.png)",
    )
    parser.add_argument(
        "--preds_dir",
        type=str,
        default=None,
        help="Directory containing prediction masks (e.g., <video_frame_id>_output_ens.jpg). Defaults depend on --solver.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="dpm",
        choices=["dpm", "nodpm"],
        help="Prediction source: dpm (default, data/out_sample_fast) or nodpm (data/out_sample)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/metrics",
        help="Directory to save metric outputs and plots",
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="Overlay transparency in [0,1] for plots")
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="overlay+error",
        choices=["overlay", "error", "overlay+error"],
        help="Visualization mode for example plots",
    )
    parser.add_argument("--plot_extremes", action="store_true", help="Plot best/mean/worst examples for the chosen metric")
    parser.add_argument(
        "--plot_metric",
        type=str,
        default="dice",
        choices=["dice", "iou", "miou"],
        help="Metric used to select best/mean/worst examples",
    )
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process this many images")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    compute(args)
