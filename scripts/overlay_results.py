''''
All credits go to https://github.com/ImprintLab/MedSegDiff/compare/master...MagnusGoltermann:MedSegDiff:master
'''
from pathlib import Path
import argparse
import os
import re
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches

def resolve_preds_dir(solver: str, preds_dir_arg: str):
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
        for sub in base.iterdir():
            if sub.is_dir() and list(sub.glob("*_output_ens.*")):
                return sub
    return base

def get_img_filename_from_videoframe_id(video_frame_id: str) -> str:
    return video_frame_id + '.png'

def get_gt_mask_filename_from_videoframe_id(video_frame_id: str) -> str:
    return video_frame_id + '.tif'

def get_pred_mask_filename_from_videoframe_id(video_frame_id: str) -> str:
    return video_frame_id + '_output_ens.jpg'

def get_img_gt_and_pred_paths_from_video_frame_id(
    video_frame_id: str,
    img_dir: str,
    gt_mask_dir: str,
    pred_mask_dir: str,
):
    img_filename = get_img_filename_from_videoframe_id(video_frame_id)
    img_path = img_dir / img_filename

    gt_filename = get_gt_mask_filename_from_videoframe_id(video_frame_id)
    gt_mask_path = gt_mask_dir / gt_filename
    
    pred_filename = get_pred_mask_filename_from_videoframe_id(video_frame_id)
    pred_mask_path = pred_mask_dir/ pred_filename

    return img_path, gt_mask_path, pred_mask_path

def extract_numeric_id(filename: str) -> Optional[str]:
    """Extract the contiguous digits id from filenames like 'ISIC_0011392.jpg'."""
    name = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(v\d+_f\d+)", name)
    return match.group(1) if match else None


def load_image_rgb(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def load_mask_binary(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load a mask (png/jpg) as binary ndarray in {0,1}. Optionally resize to (W, H)."""
    img = Image.open(path).convert("L")
    if size is not None:
        # PIL expects size as (W, H)
        img = img.resize(size, resample=Image.BILINEAR)
    arr = np.asarray(img)
    # Threshold and auto-fix polarity if mask appears inverted (lesion should be minority)
    binary = (arr >= 128).astype(np.uint8)
    if binary.mean() > 0.5:
        binary = 1 - binary
    return binary


def make_color_overlay(base: np.ndarray, mask: np.ndarray, color_rgb: Tuple[float, float, float], alpha: float, draw_contour: bool = True) -> None:
    """Plot base image and overlay a single-color mask with optional contour.
    - base: HxWx3 uint8 RGB image
    - mask: HxW {0,1}
    - color_rgb: tuple in [0,1]
    - alpha: float in [0,1]
    - draw_contour: whether to draw a crisp boundary contour
    """
    plt.imshow(base)
    color_layer = np.ones((*mask.shape, 3), dtype=np.float32)
    color_layer[..., 0] *= color_rgb[0]
    color_layer[..., 1] *= color_rgb[1]
    color_layer[..., 2] *= color_rgb[2]
    plt.imshow(color_layer, alpha=(mask.astype(np.float32) * alpha))
    if draw_contour and mask.any():
        plt.contour(mask.astype(np.float32), levels=[0.5], colors=[color_rgb], linewidths=2.0)
    plt.axis("off")


def make_error_overlay(base: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float) -> None:
    """Plot base with TP/FP/FN color overlay.
    - TP (pred=1, gt=1): green
    - FP (pred=1, gt=0): red
    - FN (pred=0, gt=1): blue
    """
    plt.imshow(base)

    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    # Combine into a color layer. Priority doesn't matter because masks are disjoint.
    h, w = gt.shape
    color_layer = np.zeros((h, w, 3), dtype=np.float32)
    # Softer, less-saturated colors for a cleaner look
    color_layer[tp] = np.array(mcolors.to_rgb("#4CAF50"), dtype=np.float32)   # soft green
    color_layer[fp] = np.array(mcolors.to_rgb("#F44336"), dtype=np.float32)   # soft red
    color_layer[fn] = np.array(mcolors.to_rgb("#3F51B5"), dtype=np.float32)   # indigo

    alpha_map = np.zeros((h, w), dtype=np.float32)
    alpha_map[tp | fp | fn] = alpha

    plt.imshow(color_layer, alpha=alpha_map)
    plt.axis("off")


def visualize_case(
    image_path: str,
    mask_path: str,
    pred_path: str,
    output_path: str,
    alpha: float = 0.4,
    mode: str = "overlay",
    suptitle: Optional[str] = None,
) -> None:
    base = load_image_rgb(image_path)  # HxWx3

    # Use mask's spatial size as reference if available, else base image size
    # PIL sizes are (W, H)
    mask_img = Image.open(mask_path).convert("L")
    mask_arr = (np.asarray(mask_img) >= 128).astype(np.uint8)
    if mask_arr.mean() > 0.5:
        mask_arr = 1 - mask_arr

    # Resize prediction to mask size
    pred_bin = load_mask_binary(pred_path, size=mask_img.size)

    # If the base image differs from the mask, resize base for aligned overlays
    h_base, w_base = base.shape[:2]
    h_mask, w_mask = mask_arr.shape
    if (w_base, h_base) != (w_mask, h_mask):
        base_resized = Image.fromarray(base).resize((w_mask, h_mask), resample=Image.BILINEAR)
        base = np.asarray(base_resized)

    # Build the figure with a variable number of panels
    panel_count = 5 if mode == "overlay+error" else 4
    fig = plt.figure(figsize=(4 * panel_count, 4), dpi=150)
    if suptitle:
        fig.suptitle(suptitle)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    def add_axis(position: int, title: str):
        ax = plt.subplot(1, panel_count, position)
        ax.set_title(title)
        ax.axis("off")
        plt.sca(ax)
        return ax

    # 1) Image
    ax1 = add_axis(1, "Image")
    ax1.imshow(base)

    # Colors (pleasant, less saturated)
    gt_color = mcolors.to_rgb("#2AA198")     # teal
    pred_color = mcolors.to_rgb("#FF8C00")   # dark orange

    # Helper to show masks as binary images
    def show_mask(mask: np.ndarray, title: str) -> None:
        plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
        plt.title(title)
        plt.axis("off")

    # 2) Ground truth mask
    add_axis(2, "Ground truth")
    show_mask(mask_arr, "Ground truth")

    # 3) Sample mask (prediction)
    add_axis(3, "Sample")
    show_mask(pred_bin, "Sample")

    next_pos = 4

    def draw_overlay_axis(position: int):
        ax = add_axis(position, "Overlay")
        # Background image + contours only (no fills)
        plt.imshow(base)
        if mask_arr.any():
            plt.contour(
                mask_arr.astype(np.float32),
                levels=[0.5],
                colors=[gt_color],
                linewidths=2.5,
            )
        if pred_bin.any():
            plt.contour(
                pred_bin.astype(np.float32),
                levels=[0.5],
                colors=[pred_color],
                linestyles=["--"],
                linewidths=2.5,
            )

        # Legend
        legend_handles = [
            mpatches.Patch(color=gt_color, label="GT"),
            mpatches.Patch(color=pred_color, label="Sample"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8)

    def draw_error_axis(position: int):
        add_axis(position, "Error map")
        make_error_overlay(base, mask_arr, pred_bin, alpha)

    if mode in {"overlay", "overlay+error"}:
        draw_overlay_axis(next_pos)
        next_pos += 1

    if mode in {"error", "overlay+error"}:
        draw_error_axis(next_pos)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def config_parser():
    parser = argparse.ArgumentParser(description="Overlay VFSS test images with GT, predictions, and error maps.")
    parser.add_argument(
        "--images_dir",
        type=str,
        default='/Users/caioseda/Documents/Trabalho/Tecgraf/projetos/MedSegDiff/vfss-data-split/data/dataset_inca/test/data/',
        help="Directory containing test images (e.g., <video_frame_id>*.png)",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default='/Users/caioseda/Documents/Trabalho/Tecgraf/projetos/MedSegDiff/vfss-data-split/data/dataset_inca/test/target/',
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
        default='data/overlay_results/',
        help="Directory to save overlay figures",
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="Overlay transparency in [0,1]")
    parser.add_argument(
        "--mode",
        type=str,
        default="overlay",
        choices=["overlay", "error", "overlay+error"],
        help="Display mode for the last panels: overlay, error, or both (overlay+error)",
    )
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process this many images")
    return parser


def main() -> None:
    parser = config_parser()
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    preds_dir = resolve_preds_dir(args.solver, args.preds_dir)
    solver_tag = args.solver

    video_frame_ids = [
        Path(file).stem
        for file in os.listdir(images_dir)
    ]

    processed = 0
    skipped = 0
    for video_frame_id in video_frame_ids:
        if args.limit and processed >= args.limit:
            break

        image_path, mask_path, pred_path = get_img_gt_and_pred_paths_from_video_frame_id(
            video_frame_id=video_frame_id,
            img_dir=images_dir,
            gt_mask_dir=masks_dir,
            pred_mask_dir=preds_dir
        )

        if not os.path.exists(mask_path) or not os.path.exists(pred_path):
            print(f"{video_frame_id} skipped. Mask or pred does not exist.")
            skipped += 1
            continue

        out_path = os.path.join(args.output_dir, args.solver, f"{video_frame_id}_{args.mode}_{solver_tag}.png")

        try:
            visualize_case(image_path, mask_path, pred_path, out_path, alpha=args.alpha, mode=args.mode, suptitle=f"{video_frame_id} [{solver_tag}]")
            processed += 1
            print(f"Saved: {out_path}")
        except Exception as e:
            skipped += 1
            print(f"Failed {video_frame_id}: {e}")

    print(f"Done. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
