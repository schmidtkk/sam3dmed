#!/usr/bin/env python3
"""Visualize a model checkpoint on a single sample.

This script loads a checkpoint (PyTorch or safetensors), builds a dummy test sample
or loads one from provided data root, runs the model, and saves visualizations.

NOTE: This script is for TESTING script mechanics with a dummy model.
      For actual inference with real models, use visualize_finetuned.py instead:
        python scripts/visualize_finetuned.py --lora_checkpoint <ckpt> --image <img> --output_dir <out>

Usage (testing only):
  python scripts/visualize_checkpoint.py --checkpoint /path/to/checkpoint.pt --allow_dummy --output_dir /tmp/vis

"""
import argparse
import os
import sys
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from loguru import logger

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('LIDRA_SKIP_INIT', '1')

from scripts.eval_medical import create_dummy_model


def safe_load_checkpoint(path: str):
    """Shareable safe loader similar to `eval_medical` to mitigate pickle risks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Checkpoint file {path} is empty")
    if path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            return load_file(path)
        except Exception:
            logger.warning("Failed to load safetensors; falling back to torch.load")

    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def create_dummy_sample():
    # shapes follow eval_medical: images (B,1,256,256), pointmap (B,256,256,3), sdf (B,256,256,1), mask (B,256,256)
    image = torch.randn(1, 1, 256, 256)
    pointmap = torch.randn(1, 256, 256, 3)
    sdf = torch.randn(1, 256, 256, 1)
    mask = (torch.rand(1, 256, 256) > 0.5).float()
    return {
        'image': image,
        'pointmap': pointmap,
        'sdf': sdf,
        'mask': mask,
    }


def overlay_mask_on_image(img: np.ndarray, mask: np.ndarray, cmap='jet'):
    import matplotlib
    from matplotlib import cm
    cmap_inst = plt.get_cmap(cmap)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        if img.shape[0] == 1:
            img_disp = img.squeeze(0)
            img_disp = img_disp / np.clip(img_disp.max(), 1e-8, None)
            img_disp = (img_disp * 255).astype(np.uint8)
            img_disp = np.stack([img_disp, img_disp, img_disp], axis=-1)
        else:
            img_disp = img.transpose(1, 2, 0)
    else:
        if img.ndim == 2:
            img_disp = np.stack([img, img, img], axis=-1)
        else:
            img_disp = img

    mask_rgb = cmap_inst(mask.astype(float))[:, :, :3]  # RGB
    alpha = 0.5
    overlay = (1 - alpha) * img_disp.astype(float) / 255.0 + alpha * mask_rgb
    overlay = np.clip(overlay, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='./vis_out')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sample_index', type=int, default=0, help='Which sample to visualize from dataset (if provided)')
    parser.add_argument('--sample_image', type=str, default=None, help='Path to image (.npy, .png) to use as sample')
    parser.add_argument('--sample_pointmap', type=str, default=None, help='Path to pointmap (.npy) to use as sample')
    parser.add_argument('--model_hf', default=None, help='HuggingFace model ID to load (e.g. facebook/sam-3d-objects/slat_mesh)')
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--visualize_format', default='both', choices=['html', 'png', 'both'], help='Which visual formats to save')
    parser.add_argument('--compare_mode', default='side_by_side', choices=['overlay', 'side_by_side'], help='Whether to overlay mask or show side-by-side')
    parser.add_argument('--allow_dummy', action='store_true', help='Allow using dummy model for testing (not for real inference)')
    args = parser.parse_args()

    logger.info('Visualization script')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    if args.model_hf is not None:
        from sam3d_objects.model.backbone.tdfy_dit.models import from_pretrained

        try:
            model = from_pretrained(args.model_hf)
            logger.info(f'Loaded model from HuggingFace: {args.model_hf}')
        except Exception as e:
            logger.error(f'Failed to load HF model {args.model_hf}: {e}')
            logger.error('Cannot proceed without a valid model. Use --allow_dummy for testing with dummy model.')
            sys.exit(1)
    elif args.allow_dummy:
        logger.warning('Using dummy model (--allow_dummy). This is for testing only and will NOT produce meaningful predictions.')
        model = create_dummy_model()
    else:
        logger.error('No model specified. Use --model_hf <hf_model_id> to load a real model.')
        logger.error('Available models: facebook/sam-3d-objects/slat_mesh, facebook/sam-3d-objects/slat_gs, etc.')
        logger.error('')
        logger.error('NOTE: For proper inference with fine-tuned models, use visualize_finetuned.py instead:')
        logger.error('  python scripts/visualize_finetuned.py --lora_checkpoint <ckpt> --image <img> --output_dir <out>')
        logger.error('')
        logger.error('If you want to test the script mechanics with a dummy model, use --allow_dummy.')
        sys.exit(1)

    # Load checkpoint
    checkpoint = None
    try:
        checkpoint = safe_load_checkpoint(args.checkpoint)
    except Exception as e:
        logger.error(f'Failed to load checkpoint: {e}')
        sys.exit(1)

    # Normalize checkpoint
    from collections.abc import Mapping
    if isinstance(checkpoint, Mapping) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        checkpoint = {'state_dict': checkpoint}

    # Inject LoRA by default
    if not args.no_lora:
        try:
            from sam3d_objects.model.lora import inject_lora, load_lora_state_dict
            inject_lora(model, target_modules=['to_qkv', 'to_out'], rank=4)
            if isinstance(checkpoint, Mapping) and 'lora_state_dict' in checkpoint:
                load_lora_state_dict(model, checkpoint['lora_state_dict'])
        except Exception as e:
            logger.warning(f'LoRA injection/loading failed: {e}')

    # If there is a full state_dict, load it and report mismatches
    if isinstance(checkpoint, Mapping) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
        key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
        ckpt_keys = set(checkpoint[key].keys())
        model_keys = set(model.state_dict().keys())
        matched = ckpt_keys & model_keys
        missing_in_model = ckpt_keys - model_keys
        missing_in_ckpt = model_keys - ckpt_keys
        logger.info(f'Checkpoint has {len(ckpt_keys)} keys, model has {len(model_keys)} keys, {len(matched)} matched.')
        if len(matched) == 0:
            logger.error('No matching keys between checkpoint and model!')
            logger.error(f'Checkpoint keys (first 10): {list(ckpt_keys)[:10]}')
            logger.error(f'Model keys (first 10): {list(model_keys)[:10]}')
            if not args.allow_dummy:
                logger.error('This likely means the checkpoint is for a different model architecture.')
                logger.error('Use --model_hf to specify the correct model, or --allow_dummy to proceed anyway.')
                sys.exit(1)
        if missing_in_model:
            logger.warning(f'{len(missing_in_model)} checkpoint keys not in model (first 5): {list(missing_in_model)[:5]}')
        if missing_in_ckpt:
            logger.warning(f'{len(missing_in_ckpt)} model keys not in checkpoint (first 5): {list(missing_in_ckpt)[:5]}')
        try:
            model.load_state_dict(checkpoint[key], strict=False)
            logger.info('Loaded checkpoint weights into model (strict=False).')
        except Exception as e:
            logger.error(f'Failed to load state dict: {e}')
            if not args.allow_dummy:
                sys.exit(1)

    model.to(args.device)
    model.eval()

    # Prepare single sample: either load from provided paths or use dummy
    if args.sample_image is not None:
        img_path = Path(args.sample_image)
        if img_path.suffix in ['.npy']:
            img = np.load(img_path)
        else:
            try:
                from PIL import Image

                im = Image.open(img_path).convert('L')
                im = im.resize((256, 256))
                arr = np.array(im)[None, ...]  # 1xHxW
                img = arr.astype(np.float32) / 255.0
            except Exception:
                logger.error('Unable to load sample image: %s', img_path)
                sys.exit(1)
        # Convert to expected shape: (1,1,256,256)
        if img.ndim == 2:
            img_tensor = torch.from_numpy(img)[None, None, ...].float()
        elif img.ndim == 3 and img.shape[0] in [1, 3]:
            img_tensor = torch.from_numpy(img)[None, ...].float()
            if img_tensor.shape[1] not in [1, 3]:
                img_tensor = img_tensor.permute(0, 3, 1, 2)
        else:
            # fallback
            img_tensor = torch.from_numpy(img)[None, None, ...].float()

        if args.sample_pointmap is not None:
            pm = np.load(args.sample_pointmap)
            pm_tensor = torch.from_numpy(pm)[None, ...].float()
        else:
            pm_tensor = torch.randn(1, 256, 256, 3)

        sample = {
            'image': img_tensor,
            'pointmap': pm_tensor,
            'sdf': torch.zeros(1, 256, 256, 1),
            'mask': torch.zeros(1, 256, 256),
        }
    else:
        sample = create_dummy_sample()

    # Run forward
    with torch.no_grad():
        image = sample['image'].to(args.device)
        pointmap = sample['pointmap'].to(args.device)
        outputs = model(image, pointmap)

    # Save SDF and overlay if possible
    sdf = outputs.get('sdf')
    vertices = outputs.get('vertices')
    faces = outputs.get('faces')

    if sdf is not None:
        pred_mask = (sdf <= 0).cpu().numpy()[0]
        img = sample['image'][0].cpu().numpy()
        # Save mask and sdf arrays
        np.save(out_dir / 'pred_sdf.npy', sdf.cpu().numpy())
        np.save(out_dir / 'pred_mask.npy', pred_mask)

        # Build visualization according to user preference
        if args.compare_mode == 'overlay':
            overlay = overlay_mask_on_image(img, pred_mask.squeeze(-1))
            plt.imsave(out_dir / 'overlay.png', overlay)
            logger.info('Saved overlay: %s', out_dir / 'overlay.png')
        else:
            # Create side-by-side comparison: original image | predicted mask as a colored image
            def to_display_img(img):
                if img.ndim == 3 and img.shape[0] in [1, 3]:
                    if img.shape[0] == 1:
                        img_disp = img.squeeze(0)
                        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
                        img_disp = (img_disp * 255).astype(np.uint8)
                        img_disp = np.stack([img_disp, img_disp, img_disp], axis=-1)
                    else:
                        img_disp = img.transpose(1, 2, 0)
                elif img.ndim == 2:
                    img_disp = np.stack([img, img, img], axis=-1)
                else:
                    img_disp = img
                return img_disp

            left = to_display_img(img)
            mask = pred_mask.squeeze(-1)
            # Map mask to RGB using colormap
            cmap = plt.get_cmap('jet')
            mask_rgb = cmap(mask)[:, :, :3]
            mask_rgb = (mask_rgb * 255).astype(np.uint8)

            # Ensure both left and right have same height and width
            if left.shape[0] != mask_rgb.shape[0] or left.shape[1] != mask_rgb.shape[1]:
                # Resize mask to match image using simple numpy repeat or re-sampling if needed
                from skimage.transform import resize
                mask_rgb = (resize(mask_rgb, left.shape[:2], preserve_range=True) ).astype(np.uint8)

            compare = np.concatenate([left, mask_rgb], axis=1)
            plt.imsave(out_dir / 'compare.png', compare)
            logger.info('Saved comparison (side-by-side): %s', out_dir / 'compare.png')
        # Save raw sdf and mask
        np.save(out_dir / 'pred_sdf.npy', sdf.cpu().numpy())
        np.save(out_dir / 'pred_mask.npy', pred_mask)

    # 3D visualization if vertices exist
    try:
        from sam3d_objects.utils.visualization import SceneVisualizer
        have_scene = True
    except Exception:
        have_scene = False

    if vertices is not None and have_scene:
        try:
            if isinstance(vertices, list):
                vert = vertices[0]
            else:
                vert = vertices[0] if vertices.dim() > 2 else vertices
            fig = SceneVisualizer.plot_scene(points_local=vert)
            html_path = out_dir / 'mesh.html'
            fig.write_html(str(html_path))
            logger.info('Saved 3D mesh html: %s', html_path)
            if args.visualize_format in ['png', 'both']:
                try:
                    png = fig.to_image(engine='kaleido')
                    with open(out_dir / 'mesh.png', 'wb') as f:
                        f.write(png)
                    logger.info('Saved mesh PNG')
                except Exception:
                    logger.warning('Failed to render mesh as PNG (kaleido not available)')
        except Exception as e:
            logger.exception(f'3D visualization failed: {e}')
    else:
        if vertices is None:
            logger.info('No vertices output from model; 3D visualization skipped')
        else:
            logger.info('SceneVisualizer not available; 3D visualization skipped')

    # Also save metadata summary
    meta = {'checkpoint': args.checkpoint, 'overlay': str(out_dir / 'overlay.png')}
    if vertices is not None:
        meta['vertices'] = str(out_dir / 'mesh.html') if have_scene else 'no-3d'
    meta_path = out_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info('Visualization complete. Results in %s', out_dir)


if __name__ == '__main__':
    main()
