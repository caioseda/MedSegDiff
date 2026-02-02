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

def _load_image_rgb(path: str, size: Optional[Tuple[int, int]] = None, resample_method=Image.BILINEAR) -> np.ndarray:
    """Load an RGB image (png/jpg) as ndarray. Optionally resize to (W, H)."""
    image = Image.open(path).convert("RGB")

    if size is not None:
        image = image.resize(size, resample=resample_method)

    return np.asarray(image)

def _load_mask_binary(path: str, size: Optional[Tuple[int, int]] = None, resample_method=Image.NEAREST, binarize: bool=True) -> np.ndarray:
    """Load a mask (png/jpg) as binary ndarray in {0,1}. Optionally resize to (W, H)."""
    mask_img = Image.open(path).convert("L")
    
    if size is not None:
        mask_img = mask_img.resize(size, resample=Image.NEAREST)
    
    # Threshold to binary
    mask_arr = np.asarray(mask_img)
    mask_arr = (mask_arr / 255.0)
    if binarize:
        mask_arr = (mask_arr >= 0.5).astype(np.uint8)

    return mask_arr

def _resolve_preds_dir(solver: str, preds_dir_arg: str):
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

class VFSSImageVisualizer:
    def __init__(self, img_dir: str, gt_mask_dir: str, pred_mask_dir: str, size: Optional[Tuple[int, int]] = (256, 256)):
        self.img_dir = Path(img_dir)
        self.gt_mask_dir = Path(gt_mask_dir)
        self.pred_mask_dir = Path(pred_mask_dir)
        self.size = size
    
    def get_paths_from_video_frame_id(self, video_frame_id: str):
        img_filename = video_frame_id + '.png'
        img_path = self.img_dir / img_filename

        gt_filename = video_frame_id + '.tif'
        gt_mask_path = self.gt_mask_dir / gt_filename
        
        pred_filename = video_frame_id + '_output_ens.jpg'
        pred_mask_path = self.pred_mask_dir/ pred_filename

        return img_path, gt_mask_path, pred_mask_path

    def draw_mask_contour(
        self,
        mask: np.ndarray,
        color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        title: str = "",
        linewidth: float = 1,
        fill: bool = False,
        alpha: float = 0.4,
    ) -> None:
        """Draw mask contours."""
        
        ax = plt.gca()
        ax.contour(
            mask.astype(np.float32),
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
            fill=fill,
            alpha=alpha,
        )
        # ax.title(title)
        # ax.axis("off")

    def draw_gt_and_pred_mask_overlay(
            self,
            gt_mask: np.array, 
            pred_mask: np.array, 
            base_img: np.array = None,
            gt_color: Tuple[float, float, float] = mcolors.to_rgb("#2AA198"), 
            pred_color: Tuple[float, float, float] = mcolors.to_rgb("#FF8C00"), 
            linewidth: float = 1, 
            alpha: float = 0.4
        ) -> None:
        """Draw mask ground truth and prediction contours."""

        # Plot base image
        if base_img is not None:
            plt.imshow(base_img)
            plt.gca().invert_yaxis()  # Ensure the y-axis matches the image orientation

        # Plot ground truth contour
        self.draw_mask_contour(
            gt_mask,
            color=gt_color,
            linewidth=linewidth,
            alpha=alpha,
        )
        
        # Plot prediction contour
        self.draw_mask_contour(
            pred_mask,
            color=pred_color,
            linewidth=linewidth,
            alpha=alpha,
        )

        # Legend
        legend_handles = [
            mpatches.Patch(color=gt_color, label="GT"),
            mpatches.Patch(color=pred_color, label="Pred"),
        ]
        plt.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8)
        plt.axis("off")

    def draw_error_overlay(
            self,
            gt_mask: np.ndarray,
            pred_mask: np.ndarray,
            base_img: np.ndarray = None,
            alpha: float=0.4,
            add_legend: bool = True
        ) -> None:
        """Plot base with TP/FP/FN color overlay.
        - TP (pred=1, gt=1): green
        - FP (pred=1, gt=0): red
        - FN (pred=0, gt=1): blue
        """
        if base_img is not None:
            plt.imshow(base_img)

        tp_filter = (pred_mask == 1) & (gt_mask == 1) # Preicted correctly
        fp_filter = (pred_mask == 1) & (gt_mask == 0) # Predicted but not in GT
        fn_filter = (pred_mask == 0) & (gt_mask == 1) # In GT but not predicted

        # Combine into a color layer. Priority doesn't matter because masks are disjoint.
        h, w = gt_mask.shape
        color_layer = np.zeros((h, w, 3), dtype=np.float32)
        
        # Set colors for each error type
        color_layer[tp_filter] = np.array(mcolors.to_rgb("#4CAF50"), dtype=np.float32)   # soft green
        color_layer[fp_filter] = np.array(mcolors.to_rgb("#F44336"), dtype=np.float32)   # soft red
        color_layer[fn_filter] = np.array(mcolors.to_rgb("#3F51B5"), dtype=np.float32)   # indigo

        # Set alpha map only where there is an error
        in_any_mask_filter = tp_filter | fp_filter | fn_filter
        alpha_map = np.zeros((h, w), dtype=np.float32)
        alpha_map[in_any_mask_filter] = alpha

        # Add legend
        if add_legend:
            ax = plt.gca()
            legend_handles = [
                mpatches.Patch(color=mcolors.to_rgb("#4CAF50"), label="TP"),
                mpatches.Patch(color=mcolors.to_rgb("#F44336"), label="FP"),
                mpatches.Patch(color=mcolors.to_rgb("#3F51B5"), label="FN"),
            ]
            ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8)

        plt.imshow(color_layer, alpha=alpha_map)
        plt.axis("off")

    def visualize_case(
        self,
        video_frame_id: str,
        output_path: str = '',
        show_masks: bool = False,
        show_error:  bool = False,
        alpha: float = 0.4,
        linewidth: float = 1,
    ) -> None:
        
        # Get paths
        image_path, mask_path, pred_path = self.get_paths_from_video_frame_id(video_frame_id)

        # Load data
        base_img = _load_image_rgb(image_path, size=self.size)  # HxWx3
        gt_mask = _load_mask_binary(mask_path, size=self.size) # HxW
        pred_mask = _load_mask_binary(pred_path) # HxW

        # # Build the figure with a variable number of panels
        # panel_count = 5 if mode == "overlay+error" else 4
        # fig = plt.figure(figsize=(4 * panel_count, 4), dpi=150)
        fig = plt.figure(figsize=(16, 4), dpi=150)

        # Plot base image
        plt.imshow(base_img)

        if show_masks:
            self.draw_gt_and_pred_mask_overlay(gt_mask, pred_mask, linewidth=linewidth, alpha=alpha)
            
        if show_error:
            self.draw_error_overlay(gt_mask, pred_mask, alpha=alpha)

        if output_path :
            print("Saving to ", output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
        
        # Set title
        plt.title(f"{video_frame_id}")
        plt.axis("off")
        plt.show()
        plt.close(fig)

    def visualize_masks(
        self,
        video_frame_id: str,
        show_gt: bool = True,
        show_pred: bool = True,
        binarize_pred: bool = True,
        contour_overlay: bool = False,
        show_image: bool = True,
        plot: bool = True,
    ) -> list:
        '''Visualize ground truth/prediction masks and return axes list
        
        Args:
            video_frame_id (str): Video frame identifier
            show_gt (bool): Whether to show ground truth mask
            show_pred (bool): Whether to show prediction mask
            binarize_pred (bool): Whether to binarize prediction mask
            contour_overlay (bool): Whether to overlay contours on image
            show_image (bool): Whether to show the base image
            plot (bool): Whether to display the plot immediately

        Returns:
            list: List of matplotlib axes with the plotted masks
        '''
        
        _, gt_mask_path, pred_mask_path = self.get_paths_from_video_frame_id(video_frame_id)
        
        axes = []
        
        if show_gt:
            gt_mask = _load_mask_binary(gt_mask_path, size=self.size)
            ax_gt = plt.subplot(1, 2, 1)
            ax_gt.set_title("Ground truth")
            ax_gt.axis("off")
            if show_image:
                ax_gt.imshow(gt_mask, cmap='gray')
            if contour_overlay:
                self.draw_mask_contour(gt_mask, color=mcolors.to_rgb("#2AA198"), alpha=1.0)
            axes.append(ax_gt)
        
        if show_pred:
            pred_mask = _load_mask_binary(pred_mask_path, size=self.size, binarize=binarize_pred)
            ax_pred = plt.subplot(1, 2, 2)
            ax_pred.set_title("Prediction")
            ax_pred.axis("off")
            if show_image:
                ax_pred.imshow(pred_mask, cmap='gray')
            if contour_overlay:
                self.draw_mask_contour(pred_mask, color=mcolors.to_rgb("#FF8C00"), alpha=1.0)
            axes.append(ax_pred)
        
        if plot:
            plt.tight_layout()
            plt.show()
        else:
            # Close the figure to avoid plotting
            plt.close()  
        
        return axes

    def visualize_comparison(
        self,
        video_frame_id: str,
        output_path: str = '',
        alpha: float = 0.7,
        linewidth: float = 1,
        show: bool = True,
    ) -> None:
        '''Visualize comparison of GT and prediction masks side by side'''
        
        # Get paths
        image_path, mask_path, pred_path = self.get_paths_from_video_frame_id(video_frame_id)

        # Load data
        base_img = _load_image_rgb(image_path, size=self.size)  # HxWx3
        gt_mask = _load_mask_binary(mask_path, size=self.size) # HxW
        pred_mask = _load_mask_binary(pred_path) # HxW

        # Build the figure with 4 panels
        fig = plt.figure(figsize=(12, 4), dpi=150)

        # 1) Ground truth mask
        ax1 = plt.subplot(1, 4, 1)
        ax1.set_title("Ground truth")
        ax1.axis("off")
        ax1.imshow(base_img)
        self.draw_mask_contour(gt_mask, color=mcolors.to_rgb("#2AA198"), linewidth=linewidth, alpha=1.0)

        # 2) Prediction mask
        ax2 = plt.subplot(1, 4, 2)
        ax2.set_title("Prediction")
        ax2.axis("off")
        ax2.imshow(base_img)
        self.draw_mask_contour(pred_mask, color=mcolors.to_rgb("#FF8C00"), linewidth=linewidth, alpha=1.0)

        # 3) Overlay
        ax3 = plt.subplot(1, 4, 3)
        ax3.set_title("Overlay")
        ax3.axis("off")
        ax3.imshow(base_img)
        self.draw_gt_and_pred_mask_overlay(gt_mask, pred_mask, linewidth=linewidth, alpha=alpha)

        # 4) Error map
        ax4 = plt.subplot(1, 4, 4)
        ax4.set_title("Error map")
        ax4.axis("off")
        self.draw_error_overlay(gt_mask, pred_mask, base_img=base_img, alpha=alpha)

        plt.tight_layout()

        if output_path:
            print("Saving to ", output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
        
        plt.suptitle(f"{video_frame_id}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def visualize_multiple_comparisons(
        self,
        video_frame_ids: list,
        output_dir: str = '',
        alpha: float = 0.7,
        linewidth: float = 1,
    ) -> None:
        '''Visualize comparison of GT and prediction masks for multiple video_frame_ids'''
        
        for video_frame_id in video_frame_ids:
            output_path = os.path.join(output_dir, f"{video_frame_id}_comparison.png") if output_dir else ''
            self.visualize_comparison(
                video_frame_id=video_frame_id,
                output_path=output_path,
                alpha=alpha,
                linewidth=linewidth,
                show=False
            )

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
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process this many images")
    return parser


def main() -> None:
    parser = config_parser()
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    preds_dir = _resolve_preds_dir(args.solver, args.preds_dir)
    solver_tag = args.solver

    video_frame_ids = [
        Path(file).stem
        for file in os.listdir(images_dir)
    ]

    visualizer = VFSSImageVisualizer(
        img_dir=images_dir,
        gt_mask_dir=masks_dir,
        pred_mask_dir=preds_dir,
        size=(256, 256)
    )

    processed = 0
    skipped = 0
    for video_frame_id in video_frame_ids:
        if args.limit and processed >= args.limit:
            break

        try:
            out_path = os.path.join(args.output_dir, args.solver, f"{video_frame_id}_{solver_tag}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            visualizer.visualize_comparison(
                video_frame_id=video_frame_id,
                output_path=out_path,
                alpha=args.alpha
            )

            processed += 1
            print(f"Saved: {out_path}")
        except Exception as e:
            skipped += 1
            print(f"Failed {video_frame_id}: {e}")

    print(f"Done. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
