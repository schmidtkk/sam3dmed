#!/usr/bin/env python3
"""Visualize a fine-tuned LoRA checkpoint using the full inference pipeline.

This script uses the official Inference class to load the pretrained models,
then injects LoRA and loads the fine-tuned weights before running inference.

Usage:
  python scripts/visualize_finetuned.py \
      --lora_checkpoint /path/to/lora.pt \
      --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
      --mask_dir notebook/images/shutterstock_stylish_kidsroom_1640806567 \
      --mask_index 14 \
      --output_dir results/visualizations_finetuned

"""
import argparse
import os
import sys
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger

# Set environment before imports
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "notebook"))


def safe_load_checkpoint(path: str):
    """Load checkpoint with validation and fallback logic."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Checkpoint file {path} is empty")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description="Visualize fine-tuned LoRA checkpoint")
    parser.add_argument(
        "--pipeline_config",
        default="checkpoints/hf/pipeline.yaml",
        help="Path to pipeline.yaml config file",
    )
    parser.add_argument(
        "--lora_checkpoint",
        required=True,
        help="Path to fine-tuned LoRA checkpoint (.pt, .ckpt, .safetensors)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (RGBA with mask in alpha, or RGB)",
    )
    parser.add_argument(
        "--mask_dir",
        default=None,
        help="Directory containing mask files (if not embedded in image alpha)",
    )
    parser.add_argument(
        "--mask_index",
        type=int,
        default=0,
        help="Index of mask to use from mask_dir",
    )
    parser.add_argument(
        "--output_dir",
        default="./vis_finetuned",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank used during fine-tuning",
    )
    parser.add_argument(
        "--lora_modules",
        nargs="+",
        default=["to_qkv", "to_out"],
        help="Module names targeted by LoRA during fine-tuning",
    )
    parser.add_argument(
        "--skip_lora",
        action="store_true",
        help="Skip LoRA injection (for baseline comparison)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model (requires GPU, slower first run)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading inference pipeline...")
    
    # Import the official Inference wrapper
    try:
        from inference import Inference, load_image, load_single_mask
    except ImportError:
        logger.error(
            "Cannot import from 'inference'. "
            "Make sure notebook/inference.py exists and is accessible."
        )
        sys.exit(1)

    # Load the pipeline
    config_path = args.pipeline_config
    if not os.path.exists(config_path):
        logger.error(f"Pipeline config not found: {config_path}")
        logger.error(
            "Download checkpoints first with: python scripts/download_hf_checkpoints.py"
        )
        sys.exit(1)

    try:
        inference = Inference(config_path, compile=args.compile)
    except Exception as e:
        logger.error(f"Failed to load inference pipeline: {e}")
        logger.exception(e)
        sys.exit(1)

    logger.info("Pipeline loaded successfully.")

    # Inject LoRA into the pipeline's models if not skipping
    if not args.skip_lora:
        logger.info(
            f"Injecting LoRA (rank={args.lora_rank}, modules={args.lora_modules})..."
        )
        try:
            from sam3d_objects.model.lora import inject_lora, load_lora_state_dict

            # The pipeline has models in inference._pipeline.models
            pipeline_models = inference._pipeline.models
            lora_injected_count = 0
            for model_name, model in pipeline_models.items():
                if model is None:
                    continue
                try:
                    inject_lora(
                        model,
                        target_modules=args.lora_modules,
                        rank=args.lora_rank,
                    )
                    lora_injected_count += 1
                    logger.debug(f"Injected LoRA into {model_name}")
                except Exception as e:
                    logger.debug(f"Could not inject LoRA into {model_name}: {e}")

            logger.info(f"Injected LoRA into {lora_injected_count} sub-models.")

            # Load LoRA weights from checkpoint
            logger.info(f"Loading LoRA weights from {args.lora_checkpoint}...")
            checkpoint = safe_load_checkpoint(args.lora_checkpoint)

            # Determine structure of checkpoint
            from collections.abc import Mapping

            if isinstance(checkpoint, Mapping):
                if "lora_state_dict" in checkpoint:
                    lora_sd = checkpoint["lora_state_dict"]
                elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    # Raw state dict - assume it's directly LoRA weights
                    lora_sd = checkpoint
                else:
                    lora_sd = None
                    logger.warning(
                        "Checkpoint does not contain 'lora_state_dict' key and is not a raw state dict."
                    )
            else:
                lora_sd = None
                logger.warning("Unexpected checkpoint format.")

            if lora_sd is not None:
                loaded_count = 0
                for model_name, model in pipeline_models.items():
                    if model is None:
                        continue
                    try:
                        load_lora_state_dict(model, lora_sd, strict=False)
                        loaded_count += 1
                        logger.debug(f"Loaded LoRA weights into {model_name}")
                    except Exception as e:
                        logger.debug(f"Could not load LoRA into {model_name}: {e}")
                logger.info(f"Loaded LoRA weights into {loaded_count} sub-models.")
            else:
                logger.warning(
                    "No LoRA state dict found in checkpoint. Proceeding with injected but uninitialized LoRA."
                )

        except ImportError as e:
            logger.error(f"LoRA module not found: {e}")
            logger.error(
                "Ensure sam3d_objects.model.lora exists. Proceeding without LoRA."
            )
        except Exception as e:
            logger.error(f"Failed to inject/load LoRA: {e}")
            logger.exception(e)
    else:
        logger.info("Skipping LoRA injection (--skip_lora).")

    # Load input image and mask
    logger.info(f"Loading image: {args.image}")
    try:
        image = load_image(args.image)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)

    mask = None
    if args.mask_dir is not None:
        logger.info(f"Loading mask from {args.mask_dir} index={args.mask_index}")
        try:
            mask = load_single_mask(args.mask_dir, index=args.mask_index)
        except Exception as e:
            logger.error(f"Failed to load mask: {e}")
            sys.exit(1)

    # Run inference
    logger.info("Running inference...")
    try:
        output = inference(image, mask, seed=args.seed)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.exception(e)
        sys.exit(1)

    logger.info("Inference complete. Saving outputs...")

    # Save outputs
    # Gaussian splat
    if "gs" in output and output["gs"] is not None:
        ply_path = out_dir / "splat.ply"
        try:
            output["gs"].save_ply(str(ply_path))
            logger.info(f"Saved Gaussian splat: {ply_path}")
        except Exception as e:
            logger.warning(f"Failed to save Gaussian splat: {e}")

    # Mesh
    if "mesh" in output and output["mesh"] is not None:
        try:
            mesh = output["mesh"]
            # Attempt to save as OBJ or PLY depending on what's available
            mesh_path = out_dir / "mesh.obj"
            if hasattr(mesh, "export"):
                mesh.export(str(mesh_path))
            elif hasattr(mesh, "save_obj"):
                mesh.save_obj(str(mesh_path))
            elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                # Manual save
                import trimesh
                tm = trimesh.Trimesh(
                    vertices=mesh.vertices.cpu().numpy(),
                    faces=mesh.faces.cpu().numpy(),
                )
                tm.export(str(mesh_path))
            logger.info(f"Saved mesh: {mesh_path}")
        except Exception as e:
            logger.warning(f"Failed to save mesh: {e}")

    # Render a 2D preview if available
    if "gs" in output and output["gs"] is not None:
        try:
            # Render the Gaussian splat to an image
            from sam3d_objects.utils.visualization import SceneVisualizer
            fig = SceneVisualizer.plot_scene(gaussians=output["gs"])
            html_path = out_dir / "scene.html"
            fig.write_html(str(html_path))
            logger.info(f"Saved 3D scene HTML: {html_path}")
        except Exception as e:
            logger.warning(f"Failed to create 3D visualization: {e}")

    # Save a simple comparison image
    try:
        from PIL import Image as PILImage
        img_pil = PILImage.open(args.image).convert("RGB")
        img_np = np.array(img_pil)

        # Create a simple output visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # For the second panel, show the mask if available
        if mask is not None:
            if isinstance(mask, np.ndarray):
                axes[1].imshow(mask, cmap="gray")
            else:
                axes[1].imshow(np.array(mask), cmap="gray")
            axes[1].set_title("Input Mask")
        else:
            axes[1].text(
                0.5, 0.5, "No separate mask\n(embedded in alpha)",
                ha="center", va="center", fontsize=12
            )
            axes[1].set_title("Mask Info")
        axes[1].axis("off")

        plt.tight_layout()
        compare_path = out_dir / "input_preview.png"
        plt.savefig(compare_path, dpi=150)
        plt.close()
        logger.info(f"Saved input preview: {compare_path}")
    except Exception as e:
        logger.warning(f"Failed to create input preview: {e}")

    # Save metadata
    metadata = {
        "pipeline_config": str(args.pipeline_config),
        "lora_checkpoint": str(args.lora_checkpoint),
        "image": str(args.image),
        "mask_dir": str(args.mask_dir) if args.mask_dir else None,
        "mask_index": args.mask_index,
        "seed": args.seed,
        "lora_rank": args.lora_rank,
        "lora_modules": args.lora_modules,
        "skip_lora": args.skip_lora,
        "outputs": {
            "splat": str(out_dir / "splat.ply") if (out_dir / "splat.ply").exists() else None,
            "mesh": str(out_dir / "mesh.obj") if (out_dir / "mesh.obj").exists() else None,
            "scene_html": str(out_dir / "scene.html") if (out_dir / "scene.html").exists() else None,
        },
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")

    logger.info(f"Visualization complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
