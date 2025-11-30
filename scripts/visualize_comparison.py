#!/usr/bin/env python3
"""Dual-mode visualization: pretrained vs fine-tuned LoRA comparison.

This script supports two modes:
  Mode 1: Pretrained model only (--mode pretrained or --no_lora)
  Mode 2: Fine-tuned with LoRA (--mode finetuned)

It can run on:
  - Natural scene images from the notebook/images directory
  - Medical (TS) images from the test data

Usage:
  # Mode 1: Pretrained only on natural scene
  python scripts/visualize_comparison.py --mode pretrained \
      --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
      --mask_dir notebook/images/shutterstock_stylish_kidsroom_1640806567 \
      --mask_index 14 \
      --output_dir results/vis_pretrained_natural

  # Mode 2: Fine-tuned LoRA on medical data
  python scripts/visualize_comparison.py --mode finetuned \
      --lora_checkpoint checkpoints/medical/best_lora.pt \
      --medical_data_dir /path/to/ts_data \
      --output_dir results/vis_finetuned_medical

  # Run both modes for comparison
  python scripts/visualize_comparison.py --mode both \
      --lora_checkpoint checkpoints/medical/best_lora.pt \
      --image notebook/images/human_object/image.png \
      --mask_dir notebook/images/human_object \
      --output_dir results/vis_comparison
"""
import argparse
import os
import sys
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Tuple
from copy import deepcopy

# Set environment before heavy imports
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from loguru import logger
from PIL import Image
import trimesh
from sam3d_objects.utils.visualization.plotly.plot_scene import plot_tdfy_scene
from sam3d_objects.utils.visualization.plotly.save_scene import img_bytes_to_np
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    PointLights,
)
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebook"))


def safe_load_checkpoint(path: str) -> Dict[str, Any]:
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


class LoRASwitch:
    """Context manager for enabling/disabling LoRA in a model."""
    
    def __init__(self, pipeline, enable: bool = True):
        """
        Args:
            pipeline: The Inference pipeline object
            enable: Whether LoRA should be enabled (True) or disabled (False)
        """
        self.pipeline = pipeline
        self.enable = enable
        self._original_scales = {}
    
    def __enter__(self):
        """Enable or disable LoRA by setting scale factors."""
        if hasattr(self.pipeline, '_pipeline') and hasattr(self.pipeline._pipeline, 'models'):
            models = self.pipeline._pipeline.models
            for model_name, model in models.items():
                if model is None:
                    continue
                self._set_lora_scale(model, model_name, 1.0 if self.enable else 0.0)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original LoRA scales."""
        if hasattr(self.pipeline, '_pipeline') and hasattr(self.pipeline._pipeline, 'models'):
            models = self.pipeline._pipeline.models
            for model_name, model in models.items():
                if model is None:
                    continue
                if model_name in self._original_scales:
                    self._restore_lora_scale(model, model_name)
        return False
    
    def _set_lora_scale(self, model: torch.nn.Module, model_name: str, scale: float):
        """Set LoRA scale for all LoRA layers in a model."""
        for name, module in model.named_modules():
            if hasattr(module, 'lora_scale'):
                key = f"{model_name}.{name}"
                if key not in self._original_scales:
                    self._original_scales[key] = module.lora_scale
                module.lora_scale = scale
            # Also handle LoRALinear style with scaling attribute
            if hasattr(module, 'scaling'):
                key = f"{model_name}.{name}.scaling"
                if key not in self._original_scales:
                    self._original_scales[key] = module.scaling
                module.scaling = scale if self.enable else 0.0
    
    def _restore_lora_scale(self, model: torch.nn.Module, model_name: str):
        """Restore original LoRA scales."""
        for name, module in model.named_modules():
            key = f"{model_name}.{name}"
            if key in self._original_scales and hasattr(module, 'lora_scale'):
                module.lora_scale = self._original_scales[key]
            key_scaling = f"{model_name}.{name}.scaling"
            if key_scaling in self._original_scales and hasattr(module, 'scaling'):
                module.scaling = self._original_scales[key_scaling]


def load_natural_scene_sample(
    image_path: str,
    mask_dir: Optional[str] = None,
    mask_index: int = 0,
) -> Tuple[Any, Any]:
    """Load a natural scene sample (image + mask)."""
    from inference import load_image, load_single_mask
    
    image = load_image(image_path)
    mask = None
    if mask_dir is not None:
        mask = load_single_mask(mask_dir, index=mask_index)
    return image, mask


def load_medical_sample(
    data_dir: str,
    sample_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Load a medical image sample from TS dataset format.
    
    Supports:
    - Raw NIfTI with subdirectories: case_dir/case-image.nii.gz + case-label.nii.gz
    - NIfTI files (*_img.nii.gz + *_mask.nii.gz)
    - nnUNet preprocessed format (*.b2nd + *_seg.b2nd)
    - NPZ files (preprocessed slices)
    - PNG images (fallback)
    
    Returns:
        Tuple of (image, mask, gt_mesh_path) as numpy arrays and optional path to GT mesh
    """
    import nibabel as nib
    data_path = Path(data_dir)
    
    # Check for TS dataset structure: subdirectories with case-image.nii.gz
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if subdirs:
        # Sort and pick sample
        subdirs = sorted(subdirs)
        case_dir = subdirs[sample_index % len(subdirs)]
        
        # Look for *-image.nii.gz and *-label.nii.gz
        img_files = list(case_dir.glob("*-image.nii.gz")) + list(case_dir.glob("*_image.nii.gz"))
        label_files = list(case_dir.glob("*-label.nii.gz")) + list(case_dir.glob("*_label.nii.gz"))
        
        if img_files and label_files:
            logger.info(f"Loading TS case: {case_dir.name}")
            img_nii = nib.load(str(img_files[0]))
            mask_nii = nib.load(str(label_files[0]))
            
            image_vol = img_nii.get_fdata()
            mask_vol = mask_nii.get_fdata()
            
            # Take middle slice (axial - last axis typically)
            if image_vol.ndim == 3:
                # Find axis with smallest dimension (likely slice axis)
                slice_axis = np.argmin(image_vol.shape)
                mid_idx = image_vol.shape[slice_axis] // 2
                
                image = np.take(image_vol, mid_idx, axis=slice_axis)
                mask = np.take(mask_vol, mid_idx, axis=slice_axis)
            else:
                image = image_vol
                mask = mask_vol
            
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
            mask = (mask > 0).astype(np.uint8)
            
            return image, mask
    
    # Try nnUNet b2nd format
    b2nd_files = sorted(data_path.glob("*.b2nd"))
    if b2nd_files:
        try:
            import blosc2
            # Filter out segmentation files
            img_files = [f for f in b2nd_files if "_seg" not in f.name]
            seg_files = [f for f in b2nd_files if "_seg" in f.name]
            
            if img_files:
                img_file = img_files[sample_index % len(img_files)]
                img_data = blosc2.open(str(img_file))[:]
                
                # Find matching segmentation
                case_id = img_file.stem
                seg_file = data_path / f"{case_id}_seg.b2nd"
                if seg_file.exists():
                    mask_data = blosc2.open(str(seg_file))[:]
                else:
                    mask_data = np.ones_like(img_data)
                
                # Take middle slice
                if img_data.ndim == 4:  # (C, D, H, W)
                    mid_slice = img_data.shape[1] // 2
                    image = img_data[0, mid_slice]
                    mask = mask_data[0, mid_slice] if mask_data.ndim == 4 else mask_data[mid_slice]
                elif img_data.ndim == 3:  # (D, H, W)
                    mid_slice = img_data.shape[0] // 2
                    image = img_data[mid_slice]
                    mask = mask_data[mid_slice] if mask_data.ndim == 3 else mask_data
                else:
                    image = img_data
                    mask = mask_data
                
                # Normalize to 0-255
                image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
                mask = (mask > 0).astype(np.uint8)
                return image, mask
        except ImportError:
            logger.debug("blosc2 not installed, skipping b2nd format")
        except Exception as e:
            logger.debug(f"Failed to load b2nd: {e}")
    
    # Look for NIfTI files directly in the directory
    img_files = sorted(data_path.glob("*_img.nii.gz")) + sorted(data_path.glob("*_image.nii.gz"))
    mask_files = sorted(data_path.glob("*_mask.nii.gz")) + sorted(data_path.glob("*_label.nii.gz"))
    
    # Also check gt_segmentations subfolder (nnUNet structure)
    gt_seg_dir = data_path / "gt_segmentations"
    if gt_seg_dir.exists():
        mask_files.extend(sorted(gt_seg_dir.glob("*.nii.gz")))
    
    if not img_files:
        # Try looking for preprocessed npz files (SAM3D format)
        npz_files = sorted(data_path.glob("*.npz"))
        if npz_files:
            npz_path = npz_files[sample_index % len(npz_files)]
            logger.info(f"Loading preprocessed NPZ: {npz_path.name}")
            data = np.load(npz_path, allow_pickle=True)
            image = data.get('image', data.get('img', None))
            mask = data.get('mask', data.get('label', data.get('seg', None)))
            
            # Find GT mesh path from case_id
            gt_mesh_path = None
            case_id = npz_path.stem.split('_axis')[0]  # e.g., 's0015.nii' from 's0015.nii_axis2_slice0177'
            mesh_dir = data_path / "meshes"
            if mesh_dir.exists():
                # Look for class1 mesh (primary segmentation)
                candidate_mesh = mesh_dir / f"{case_id}_class1.obj"
                if candidate_mesh.exists():
                    gt_mesh_path = str(candidate_mesh)
                    logger.info(f"Found GT mesh: {gt_mesh_path}")
            
            if image is not None and mask is not None:
                # SAM3D format: image is (3, H, W), mask is (H, W)
                if image.ndim == 3 and image.shape[0] == 3:
                    # Take first channel (grayscale) 
                    image = image[0]
                elif image.ndim == 3:
                    # Take middle slice if 3D volume
                    mid_slice = image.shape[2] // 2
                    image = image[:, :, mid_slice]
                    mask = mask[:, :, mid_slice] if mask.ndim == 3 else mask
                # Normalize to 0-255
                if image.max() > 1.0:
                    image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
                else:
                    image = (image * 255).astype(np.uint8)
                mask = (mask > 0).astype(np.uint8)
                return image, mask, gt_mesh_path
        
        # Fall back to any PNG images
        png_files = sorted(data_path.glob("*.png"))
        if png_files:
            img_path = png_files[sample_index % len(png_files)]
            image = np.array(Image.open(img_path).convert('L'))
            mask = np.ones_like(image, dtype=np.uint8)  # Default full mask
            return image, mask, None
        
        # Generate synthetic medical-like data for testing
        logger.warning(f"No medical image files found in {data_dir}, generating synthetic data")
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        # Add some structure
        y, x = np.ogrid[:256, :256]
        center_mask = ((x - 128)**2 + (y - 128)**2 < 50**2).astype(np.uint8)
        image = (image * 0.3 + center_mask * 200 * 0.7).astype(np.uint8)
        mask = center_mask
        return image, mask, None
    
    # Load NIfTI
    img_file = img_files[sample_index % len(img_files)]
    img_nii = nib.load(str(img_file))
    image_vol = img_nii.get_fdata()
    
    # Find matching mask
    mask_vol = None
    case_id = img_file.stem.replace('_img', '').replace('_image', '')
    for mf in mask_files:
        if case_id in mf.stem:
            mask_vol = nib.load(str(mf)).get_fdata()
            break
    
    if mask_vol is None and mask_files:
        mask_vol = nib.load(str(mask_files[0])).get_fdata()
    
    # Take middle slice
    if image_vol.ndim == 3:
        mid_z = image_vol.shape[2] // 2
        image = image_vol[:, :, mid_z]
        mask = mask_vol[:, :, mid_z] if mask_vol is not None else np.ones_like(image)
    else:
        image = image_vol
        mask = mask_vol if mask_vol is not None else np.ones_like(image)
    
    # Normalize to 0-255 range
    image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)
    
    return image, mask, None  # No GT mesh for raw NIfTI (not preprocessed)


def convert_medical_to_rgba(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Convert medical grayscale image + mask to RGBA format expected by Inference."""
    # Ensure 2D
    if image.ndim > 2:
        image = image.squeeze()
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    # Normalize to 0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Create RGB from grayscale
    rgb = np.stack([image, image, image], axis=-1)
    
    # Create alpha from mask
    alpha = (mask > 0).astype(np.uint8) * 255
    
    # Combine to RGBA
    rgba = np.concatenate([rgb, alpha[:, :, None]], axis=-1)
    
    return Image.fromarray(rgba, mode='RGBA')


def run_inference(
    pipeline,
    image,
    mask,
    seed: int = 42,
    lora_enabled: bool = True,
) -> Dict[str, Any]:
    """Run inference with optional LoRA switch."""
    with LoRASwitch(pipeline, enable=lora_enabled):
        try:
            output = pipeline(image, mask, seed=seed)
            return output
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {}


def create_comparison_figure(
    image_np: np.ndarray,
    mask_np: Optional[np.ndarray],
    output_pretrained: Dict[str, Any],
    output_finetuned: Optional[Dict[str, Any]],
    title: str = "Model Comparison",
    gt_mesh_path: Optional[str] = None,
) -> plt.Figure:
    """Create a comparison figure showing input, GT mesh, and model outputs.
    
    Layout:
    - Column 1: Input image with mask overlay
    - Column 2: GT mesh (if available)
    - Column 3: Pretrained model output
    - Column 4: Fine-tuned model output (if available)
    """
    # Determine number of columns
    n_cols = 2  # Input + Pretrained
    if gt_mesh_path is not None:
        n_cols += 1
    if output_finetuned:
        n_cols += 1
    
    fig = plt.figure(figsize=(6 * n_cols, 6))
    gs = GridSpec(1, n_cols, figure=fig)
    col_idx = 0
    
    # Input image
    ax1 = fig.add_subplot(gs[0, col_idx])
    if image_np.ndim == 2:
        ax1.imshow(image_np, cmap='gray')
    else:
        ax1.imshow(image_np)
    if mask_np is not None:
        ax1.contour(mask_np, colors='red', linewidths=1, alpha=0.7)
    ax1.set_title("Input Image + Mask")
    ax1.axis('off')
    col_idx += 1
    
    # GT mesh (if available)
    if gt_mesh_path is not None:
        ax_gt = fig.add_subplot(gs[0, col_idx])
        ax_gt.set_title("GT Mesh (from Volume)")
        if os.path.exists(gt_mesh_path):
            try:
                mesh_img = render_mesh_file_to_image(gt_mesh_path, height=512)
                ax_gt.imshow(mesh_img)
            except Exception as e:
                logger.debug(f"GT mesh render failed: {e}")
                ax_gt.text(0.5, 0.5, f"Failed to render\n{e}", ha='center', va='center', fontsize=10)
        else:
            ax_gt.text(0.5, 0.5, "GT mesh not found", ha='center', va='center', fontsize=12)
        ax_gt.axis('off')
        col_idx += 1
    
    # Pretrained output
    ax2 = fig.add_subplot(gs[0, col_idx])
    ax2.set_title("Pretrained Model Output")
    if 'mesh' in output_pretrained and output_pretrained['mesh'] is not None:
        try:
            mesh_path = output_pretrained['mesh']
            mesh_img = render_mesh_file_to_image(mesh_path, height=512)
            ax2.imshow(mesh_img)
        except Exception as e:
            logger.debug(f"Mesh render failed: {e}")
            ax2.text(0.5, 0.5, "Mesh generated\n(see 3D files)", ha='center', va='center', fontsize=12)
    elif 'gs' in output_pretrained and output_pretrained['gs'] is not None:
        ax2.text(0.5, 0.5, "Gaussian splat\ngenerated", ha='center', va='center', fontsize=12)
    else:
        ax2.text(0.5, 0.5, "No output", ha='center', va='center', fontsize=12)
    ax2.axis('off')
    col_idx += 1
    
    # Finetuned output (if available)
    if output_finetuned:
        ax3 = fig.add_subplot(gs[0, col_idx])
        ax3.set_title("Fine-tuned (LoRA) Output")
        if 'mesh' in output_finetuned and output_finetuned['mesh'] is not None:
            try:
                mesh_path = output_finetuned['mesh']
                mesh_img = render_mesh_file_to_image(mesh_path, height=512)
                ax3.imshow(mesh_img)
            except Exception as e:
                logger.debug(f"Mesh render failed: {e}")
                ax3.text(0.5, 0.5, "Mesh generated\n(see 3D files)", ha='center', va='center', fontsize=12)
        elif 'gs' in output_finetuned and output_finetuned['gs'] is not None:
            ax3.text(0.5, 0.5, "Gaussian splat\ngenerated", ha='center', va='center', fontsize=12)
        else:
            ax3.text(0.5, 0.5, "No output", ha='center', va='center', fontsize=12)
        ax3.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def render_mesh_file_to_image(mesh_file: str, height: int = 512) -> np.ndarray:
    """Render a mesh file (obj/ply) to an image using plot_tdfy_scene and return a numpy array.
    Uses plotly -> kaleido (`fig.to_image`) via img_bytes_to_np.
    """

    # Determine type: path string | trimesh.Trimesh | pytorch3d Meshes | list
    tm = None
    if isinstance(mesh_file, str):
        # Load mesh via file path
        tm = trimesh.load(mesh_file, process=False)
    elif isinstance(mesh_file, trimesh.Trimesh):
        tm = mesh_file
    elif isinstance(mesh_file, trimesh.Scene):
        # Convert scene to single mesh
        geom_list = [g for g in mesh_file.geometry.values()]
        if not geom_list:
            raise ValueError('Empty Scene')
        tm = trimesh.util.concatenate(geom_list)
    elif isinstance(mesh_file, Meshes):
        verts = mesh_file.verts_list()[0].cpu().numpy()
        faces = mesh_file.faces_list()[0].cpu().numpy()
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    elif isinstance(mesh_file, list):
        # If list of strings -> load first
        if all(isinstance(x, str) for x in mesh_file):
            tm = trimesh.load(mesh_file[0], process=False)
        else:
            # Build list of trimesh.Trimesh
            trimesh_list = []
            for m in mesh_file:
                if isinstance(m, trimesh.Trimesh):
                    trimesh_list.append(m)
                elif isinstance(m, Meshes):
                    verts = m.verts_list()[0].cpu().numpy()
                    faces = m.faces_list()[0].cpu().numpy()
                    trimesh_list.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
                elif isinstance(m, str):
                    trimesh_list.append(trimesh.load(m, process=False))
                elif hasattr(m, 'vertices') and hasattr(m, 'faces'):
                    # Generic mesh-like object with vertices/faces attributes
                    try:
                        verts = np.asarray(m.vertices)
                        faces = np.asarray(m.faces)
                        trimesh_list.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
                    except Exception:
                        continue
                elif hasattr(m, 'points') and hasattr(m, 'cells'):
                    # pyvista / similar mesh-like object
                    try:
                        verts = np.asarray(m.points)
                        # cells might be stored as flat; try to reshape if needed
                        cells = np.asarray(m.cells)
                        # If cells is flat, try to split into triangles
                        if cells.ndim == 1:
                            # pyvista default: cells: [n, v0, v1, v2, n, v0, v1, v2,...]
                            # Parse into list of tris
                            tris = []
                            i = 0
                            while i < len(cells):
                                n = int(cells[i])
                                if n == 3:
                                    tris.append([cells[i+1], cells[i+2], cells[i+3]])
                                i += n + 1
                            faces = np.array(tris)
                        else:
                            faces = cells
                        trimesh_list.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
                    except Exception:
                        continue
            if len(trimesh_list) == 0:
                raise ValueError('Unsupported mesh list content')
            if len(trimesh_list) == 1:
                tm = trimesh_list[0]
            else:
                tm = trimesh.util.concatenate(trimesh_list)
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh_file)}")

    # Center mesh at origin and normalize scale to fit in [-1, 1]
    verts_np = tm.vertices - tm.centroid
    max_extent = np.abs(verts_np).max()
    if max_extent > 0:
        verts_np = verts_np / max_extent  # Now in [-1, 1]
    
    verts = torch.tensor(verts_np.astype(np.float32))
    faces = torch.tensor(tm.faces.astype(np.int64))

    if hasattr(tm, 'visual') and hasattr(tm.visual, 'vertex_colors') and tm.visual.vertex_colors is not None:
        vc = tm.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        vertex_colors = torch.tensor(vc)
    else:
        # Use a pleasant gray color for untextured meshes
        vertex_colors = torch.ones((verts.shape[0], 3), dtype=torch.float32) * 0.7

    # Render using PyTorch3D
    try:
        from pytorch3d.renderer import FoVPerspectiveCameras
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        verts_t = verts.to(device)
        faces_t = faces.to(device)
        vertex_colors_t = vertex_colors.to(device)
        textures = TexturesVertex([vertex_colors_t.float()])
        mesh_p3d = Meshes(verts=[verts_t], faces=[faces_t], textures=textures)

        # Camera: use FoV camera looking at centered mesh from a good angle
        R, T = look_at_view_transform(dist=3.0, elev=15, azim=135)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=50)

        # Use bin_size=0 for naive rasterization (avoids overflow with dense meshes)
        raster_settings = RasterizationSettings(
            image_size=height, 
            blur_radius=0.0, 
            faces_per_pixel=1,
            bin_size=0
        )
        lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])
        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
        renderer = MeshRenderer(rasterizer, shader)
        
        images = renderer(mesh_p3d)
        img_np = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        return img_np
    except Exception as e:
        logger.debug(f"PyTorch3D render failed: {e}, falling back to plotly")
        # Fall back to plotly/kaleido
        textures = TexturesVertex([vertex_colors])
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        fig = plot_tdfy_scene({"Mesh": {"mesh": mesh}}, height=height)
        img = fig.to_image(engine="kaleido")
        img_np = img_bytes_to_np(img)
        return img_np


def inject_and_load_lora(
    pipeline,
    lora_checkpoint: str,
    lora_rank: int = 4,
    lora_modules: List[str] = None,
) -> int:
    """Inject LoRA layers and load weights from checkpoint."""
    if lora_modules is None:
        lora_modules = ["to_qkv", "to_out"]
    
    from sam3d_objects.model.lora import inject_lora, load_lora_state_dict
    
    models = pipeline._pipeline.models
    injected_count = 0
    
    # Inject LoRA into all sub-models
    for model_name, model in models.items():
        if model is None:
            continue
        try:
            inject_lora(model, target_modules=lora_modules, rank=lora_rank)
            injected_count += 1
            logger.debug(f"Injected LoRA into {model_name}")
        except Exception as e:
            logger.debug(f"Could not inject LoRA into {model_name}: {e}")
    
    logger.info(f"Injected LoRA into {injected_count} sub-models")
    
    # Load LoRA weights
    checkpoint = safe_load_checkpoint(lora_checkpoint)
    
    from collections.abc import Mapping
    if isinstance(checkpoint, Mapping):
        if "lora_state_dict" in checkpoint:
            lora_sd = checkpoint["lora_state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            lora_sd = checkpoint
        else:
            lora_sd = None
            logger.warning("No lora_state_dict found in checkpoint")
    else:
        lora_sd = None
    
    if lora_sd is not None:
        loaded_count = 0
        for model_name, model in models.items():
            if model is None:
                continue
            try:
                load_lora_state_dict(model, lora_sd, strict=False)
                loaded_count += 1
            except Exception as e:
                logger.debug(f"Could not load LoRA into {model_name}: {e}")
        logger.info(f"Loaded LoRA weights into {loaded_count} sub-models")
    
    return injected_count


def save_outputs(
    output: Dict[str, Any],
    output_dir: Path,
    prefix: str = "",
) -> Dict[str, str]:
    """Save inference outputs to files."""
    saved_files = {}
    
    # Gaussian splat
    if "gs" in output and output["gs"] is not None:
        ply_path = output_dir / f"{prefix}splat.ply"
        try:
            output["gs"].save_ply(str(ply_path))
            saved_files["splat"] = str(ply_path)
            logger.info(f"Saved: {ply_path}")
        except Exception as e:
            logger.warning(f"Failed to save Gaussian splat: {e}")
    
    # Mesh
    if "mesh" in output and output["mesh"] is not None:
        mesh = output["mesh"]
        # Support single mesh or list of meshes
        if isinstance(mesh, list):
            saved_files["mesh"] = []
            for i, m in enumerate(mesh):
                mesh_path = output_dir / f"{prefix}mesh_{i}.obj"
                try:
                    if hasattr(m, "export"):
                        m.export(str(mesh_path))
                    elif hasattr(m, "save_obj"):
                        m.save_obj(str(mesh_path))
                    else:
                        import trimesh
                        tm = trimesh.Trimesh(
                            vertices=m.vertices.cpu().numpy() if hasattr(m.vertices, 'cpu') else m.vertices,
                            faces=m.faces.cpu().numpy() if hasattr(m.faces, 'cpu') else m.faces,
                        )
                        tm.export(str(mesh_path))
                    saved_files["mesh"].append(str(mesh_path))
                    logger.info(f"Saved: {mesh_path}")
                except Exception as e:
                    logger.warning(f"Failed to save mesh element {i}: {e}")
        else:
            mesh_path = output_dir / f"{prefix}mesh.obj"
            try:
                if hasattr(mesh, "export"):
                    mesh.export(str(mesh_path))
                elif hasattr(mesh, "save_obj"):
                    mesh.save_obj(str(mesh_path))
                else:
                    import trimesh
                    tm = trimesh.Trimesh(
                        vertices=mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices,
                        faces=mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces,
                    )
                    tm.export(str(mesh_path))
                saved_files["mesh"] = str(mesh_path)
                logger.info(f"Saved: {mesh_path}")
            except Exception as e:
                logger.warning(f"Failed to save mesh: {e}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Dual-mode visualization: pretrained vs fine-tuned LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pretrained only on natural scene
  python scripts/visualize_comparison.py --mode pretrained \\
      --image notebook/images/human_object/image.png \\
      --mask_dir notebook/images/human_object

  # Fine-tuned on medical data  
  python scripts/visualize_comparison.py --mode finetuned \\
      --lora_checkpoint checkpoints/medical/best.pt \\
      --medical_data_dir /path/to/ts_data

  # Compare both modes
  python scripts/visualize_comparison.py --mode both \\
      --lora_checkpoint checkpoints/medical/best.pt \\
      --image notebook/images/human_object/image.png
        """,
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["pretrained", "finetuned", "both"],
        default="pretrained",
        help="Visualization mode: pretrained, finetuned, or both for comparison",
    )
    
    # Pipeline config
    parser.add_argument(
        "--pipeline_config",
        default="checkpoints/hf/pipeline.yaml",
        help="Path to pipeline.yaml config file",
    )
    
    # LoRA settings
    parser.add_argument(
        "--lora_checkpoint",
        default=None,
        help="Path to fine-tuned LoRA checkpoint (required for finetuned/both modes)",
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
        help="Module names targeted by LoRA",
    )
    
    # Natural scene input
    parser.add_argument(
        "--image",
        default=None,
        help="Path to natural scene image (PNG/JPG)",
    )
    parser.add_argument(
        "--mask_dir",
        default=None,
        help="Directory containing mask files for natural scene",
    )
    parser.add_argument(
        "--mask_index",
        type=int,
        default=0,
        help="Index of mask to use from mask_dir",
    )
    
    # Medical input
    parser.add_argument(
        "--medical_data_dir",
        default=None,
        help="Directory containing medical (TS) data",
    )
    parser.add_argument(
        "--medical_sample_index",
        type=int,
        default=0,
        help="Sample index for medical data",
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        default="./results/vis_comparison",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model (requires GPU)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: test loading and preprocessing only, without full inference",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ["finetuned", "both"] and args.lora_checkpoint is None:
        parser.error("--lora_checkpoint is required for finetuned/both modes")
    
    if args.image is None and args.medical_data_dir is None:
        parser.error("Either --image or --medical_data_dir must be provided")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Dry run mode: test data loading only
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("Testing data loading and preprocessing only (no full inference)")
        
        # Test natural scene loading
        if args.image is not None:
            logger.info(f"Testing natural scene loading: {args.image}")
            try:
                # Use PIL directly to avoid pytorch3d dependency
                img = Image.open(args.image)
                logger.info(f"  Image size: {img.size}, mode: {img.mode}")
                
                if args.mask_dir:
                    mask_path = Path(args.mask_dir) / f"{args.mask_index}.png"
                    if mask_path.exists():
                        mask = Image.open(mask_path)
                        logger.info(f"  Mask loaded: {mask.size}")
                    else:
                        logger.info(f"  Mask file not found: {mask_path}")
                
                # Save preview
                img_np = np.array(img.convert('RGB'))
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(img_np)
                ax.set_title(f"Natural Scene: {Path(args.image).parent.name}")
                ax.axis('off')
                preview_path = out_dir / "natural_preview.png"
                fig.savefig(preview_path, dpi=150)
                plt.close(fig)
                logger.info(f"  Saved preview: {preview_path}")
            except Exception as e:
                logger.error(f"  Failed: {e}")
        
        # Test medical data loading
        if args.medical_data_dir is not None:
            logger.info(f"Testing medical data loading: {args.medical_data_dir}")
            try:
                med_image, med_mask, gt_mesh_path = load_medical_sample(
                    args.medical_data_dir,
                    args.medical_sample_index,
                )
                logger.info(f"  Image shape: {med_image.shape}, dtype: {med_image.dtype}")
                logger.info(f"  Mask shape: {med_mask.shape}, unique values: {np.unique(med_mask)}")
                logger.info(f"  GT mesh path: {gt_mesh_path}")
                
                # Save preview
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(med_image, cmap='gray')
                axes[0].set_title("Medical Image")
                axes[0].axis('off')
                axes[1].imshow(med_mask, cmap='jet')
                axes[1].set_title("Mask")
                axes[1].axis('off')
                preview_path = out_dir / "medical_preview.png"
                fig.savefig(preview_path, dpi=150)
                plt.close(fig)
                logger.info(f"  Saved preview: {preview_path}")
            except Exception as e:
                logger.error(f"  Failed: {e}")
                logger.exception(e)
        
        # Test checkpoint loading
        if args.lora_checkpoint:
            logger.info(f"Testing LoRA checkpoint loading: {args.lora_checkpoint}")
            try:
                ckpt = safe_load_checkpoint(args.lora_checkpoint)
                logger.info(f"  Checkpoint keys: {list(ckpt.keys())[:10]}...")
                logger.info(f"  Total keys: {len(ckpt)}")
            except Exception as e:
                logger.error(f"  Failed: {e}")
        
        # Test pipeline config
        logger.info(f"Testing pipeline config: {args.pipeline_config}")
        if os.path.exists(args.pipeline_config):
            logger.info("  Config file exists")
        else:
            logger.error("  Config file NOT found")
            logger.info("  Run: python scripts/download_hf_checkpoints.py")
        
        logger.info("=== DRY RUN COMPLETE ===")
        return
    
    # Load pipeline
    logger.info("Loading inference pipeline...")
    config_path = args.pipeline_config
    if not os.path.exists(config_path):
        logger.error(f"Pipeline config not found: {config_path}")
        logger.error("Run: python scripts/download_hf_checkpoints.py first")
        sys.exit(1)
    
    try:
        from inference import Inference
        pipeline = Inference(config_path, compile=args.compile)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.exception(e)
        sys.exit(1)
    
    logger.info("Pipeline loaded successfully")
    
    # Inject LoRA if needed
    if args.mode in ["finetuned", "both"]:
        logger.info("Injecting LoRA...")
        try:
            inject_and_load_lora(
                pipeline,
                args.lora_checkpoint,
                lora_rank=args.lora_rank,
                lora_modules=args.lora_modules,
            )
        except Exception as e:
            logger.error(f"Failed to inject/load LoRA: {e}")
            if args.mode == "finetuned":
                sys.exit(1)
            logger.warning("Continuing with pretrained mode only")
            args.mode = "pretrained"
    
    # Collect samples to process
    samples = []
    
    # Natural scene sample
    if args.image is not None:
        logger.info(f"Loading natural scene: {args.image}")
        try:
            image, mask = load_natural_scene_sample(
                args.image,
                args.mask_dir,
                args.mask_index,
            )
            samples.append({
                "type": "natural",
                "name": Path(args.image).parent.name,
                "image": image,
                "mask": mask,
                "image_np": np.array(image.convert('RGB') if hasattr(image, 'convert') else image),
                "mask_np": np.array(mask) if mask is not None else None,
            })
        except Exception as e:
            logger.error(f"Failed to load natural scene: {e}")
    
    # Medical sample
    if args.medical_data_dir is not None:
        logger.info(f"Loading medical data: {args.medical_data_dir}")
        try:
            med_image, med_mask, gt_mesh_path = load_medical_sample(
                args.medical_data_dir,
                args.medical_sample_index,
            )
            # Convert to format expected by pipeline
            rgba_image = convert_medical_to_rgba(med_image, med_mask)
            samples.append({
                "type": "medical",
                "name": f"medical_sample_{args.medical_sample_index}",
                "image": rgba_image,
                "mask": None,  # Embedded in alpha
                "image_np": med_image,
                "mask_np": med_mask,
                "gt_mesh_path": gt_mesh_path,  # GT mesh from volume
            })
        except Exception as e:
            logger.error(f"Failed to load medical data: {e}")
            logger.exception(e)
    
    if not samples:
        logger.error("No samples to process")
        sys.exit(1)
    
    # Process each sample
    results = []
    for sample in samples:
        logger.info(f"Processing {sample['type']} sample: {sample['name']}")
        
        sample_dir = out_dir / sample["name"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "name": sample["name"],
            "type": sample["type"],
        }
        
        # Run pretrained (LoRA disabled)
        if args.mode in ["pretrained", "both"]:
            logger.info("Running pretrained model (LoRA OFF)...")
            output_pretrained = run_inference(
                pipeline,
                sample["image"],
                sample["mask"],
                seed=args.seed,
                lora_enabled=False,
            )
            result["pretrained"] = save_outputs(output_pretrained, sample_dir, prefix="pretrained_")
        else:
            output_pretrained = {}
        
        # Run fine-tuned (LoRA enabled)
        if args.mode in ["finetuned", "both"]:
            logger.info("Running fine-tuned model (LoRA ON)...")
            output_finetuned = run_inference(
                pipeline,
                sample["image"],
                sample["mask"],
                seed=args.seed,
                lora_enabled=True,
            )
            result["finetuned"] = save_outputs(output_finetuned, sample_dir, prefix="finetuned_")
        else:
            output_finetuned = None
        
        # Create comparison figure
        # Prefer using saved outputs (file paths) for rendering if available
        pretrained_saved = result.get("pretrained", {})
        finetuned_saved = result.get("finetuned", None)
        gt_mesh_path = sample.get("gt_mesh_path", None)
        fig = create_comparison_figure(
            sample["image_np"],
            sample["mask_np"],
            pretrained_saved,
            finetuned_saved,
            title=f"{sample['type'].capitalize()} Sample: {sample['name']}",
            gt_mesh_path=gt_mesh_path,
        )
        fig_path = sample_dir / "comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        result["comparison_figure"] = str(fig_path)
        logger.info(f"Saved comparison figure: {fig_path}")
        
        # Save input preview
        fig_input, ax = plt.subplots(1, 1, figsize=(8, 8))
        if sample["image_np"].ndim == 2:
            ax.imshow(sample["image_np"], cmap='gray')
        else:
            ax.imshow(sample["image_np"])
        if sample["mask_np"] is not None:
            ax.contour(sample["mask_np"], colors='red', linewidths=2)
        ax.set_title(f"Input: {sample['name']}")
        ax.axis('off')
        input_path = sample_dir / "input.png"
        fig_input.savefig(input_path, dpi=150, bbox_inches='tight')
        plt.close(fig_input)
        result["input_preview"] = str(input_path)
        
        results.append(result)
    
    # Save summary
    summary = {
        "mode": args.mode,
        "pipeline_config": args.pipeline_config,
        "lora_checkpoint": args.lora_checkpoint,
        "lora_rank": args.lora_rank,
        "lora_modules": args.lora_modules,
        "seed": args.seed,
        "samples": results,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")
    
    logger.info(f"Visualization complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
