# Copyright (c) Meta Platforms, Inc. and affiliates.
import warnings
import torch
from loguru import logger
from dataclasses import dataclass
from typing import Callable, Optional
import warnings

from .img_and_mask_transforms import (
    SSIPointmapNormalizer,
)


# Load and process data
@dataclass
class PreProcessor:
    """
    Preprocessor configuration for image, mask, and pointmap transforms.

    Transform application order:
    1. Pointmap normalization (if normalize_pointmap=True)
    2. Joint transforms (img_mask_pointmap_joint_transform or img_mask_joint_transform)
    3. Individual transforms (img_transform, mask_transform, pointmap_transform)

    For backward compatibility, img_mask_joint_transform is preserved. When both
    img_mask_pointmap_joint_transform and img_mask_joint_transform are present,
    img_mask_pointmap_joint_transform takes priority.
    """

    img_transform: Callable = (None,)
    mask_transform: Callable = (None,)
    img_mask_joint_transform: list[Callable] = (None,)
    rgb_img_mask_joint_transform: list[Callable] = (None,)

    # New fields for pointmap support
    pointmap_transform: Callable = (None,)
    img_mask_pointmap_joint_transform: list[Callable] = (None,)
    
    # Pointmap normalization option
    normalize_pointmap: bool = False
    pointmap_normalizer: Optional[Callable] = None
    rgb_pointmap_normalizer: Optional[Callable] = None
    preprocess_crop_size: tuple[int, int] = (256, 256)

    def __post_init__(self):
        if self.pointmap_normalizer is None:
            self.pointmap_normalizer = SSIPointmapNormalizer()
            if self.normalize_pointmap == False:
                warnings.warn("normalize_pointmap is also set to False, which means we will return the moments but not normalize the pointmap. This supports old unnormalized pointmap models, but this is dangerous behavior.", DeprecationWarning, stacklevel=2)

        if self.rgb_pointmap_normalizer is None:
            logger.warning("No rgb pointmap normalizer provided, using scale + shift ")
            self.rgb_pointmap_normalizer = self.pointmap_normalizer

        # Ensure a CropOrPad joint transform exists to enforce consistent output size
        try:
            import torch.nn.functional as F
        except Exception:
            F = None

        def _crop_or_pad_joint(img, mask, pointmap=None):
            """Center crop or pad the image/mask/pointmap to self.preprocess_crop_size.

            Pads image and mask with zeros; pad pointmap with NaNs.
            """
            th, tw = self.preprocess_crop_size
            # image: (C,H,W)
            if img is not None:
                c, h, w = img.shape
                # crop
                if h > th:
                    top = (h - th) // 2
                    img = img[:, top : top + th, :]
                if w > tw:
                    left = (w - tw) // 2
                    img = img[:, :, left : left + tw]
                # pad
                if h < th or w < tw:
                    ph = th - img.shape[1]
                    pw = tw - img.shape[2]
                    left = pw // 2
                    right = pw - left
                    top = ph // 2
                    bottom = ph - top
                    if F is not None:
                        img = F.pad(img, (left, right, top, bottom), value=0.0)
                    else:
                        pad_tensor = torch.zeros((c, th, tw), dtype=img.dtype)
                        pad_tensor[:, top : top + img.shape[1], left : left + img.shape[2]] = img
                        img = pad_tensor
            # mask: (H,W)
            if mask is not None:
                if mask.ndim == 3:
                    mask = mask[0]
                h, w = mask.shape
                if h > th:
                    top = (h - th) // 2
                    mask = mask[top : top + th, :]
                if w > tw:
                    left = (w - tw) // 2
                    mask = mask[:, left : left + tw]
                if h < th or w < tw:
                    ph = th - mask.shape[0]
                    pw = tw - mask.shape[1]
                    left = pw // 2
                    top = ph // 2
                    if F is not None:
                        mask = F.pad(mask.unsqueeze(0), (left, pw - left, top, ph - top), value=0.0).squeeze(0)
                    else:
                        pad_mask = torch.zeros((th, tw), dtype=mask.dtype)
                        pad_mask[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
                        mask = pad_mask
            # pointmap: (C,H,W) or None - pad with NaNs
            if pointmap is not None:
                ch, h, w = pointmap.shape
                if h > th:
                    top = (h - th) // 2
                    pointmap = pointmap[:, top : top + th, :]
                if w > tw:
                    left = (w - tw) // 2
                    pointmap = pointmap[:, :, left : left + tw]
                if h < th or w < tw:
                    ph = th - pointmap.shape[1]
                    pw = tw - pointmap.shape[2]
                    left = pw // 2
                    right = pw - left
                    top = ph // 2
                    bottom = ph - top
                    if F is not None:
                        pointmap = F.pad(pointmap, (left, right, top, bottom), value=float("nan"))
                    else:
                        pad_pm = torch.full((ch, th, tw), float("nan"), dtype=pointmap.dtype)
                        pad_pm[:, top : top + pointmap.shape[1], left : left + pointmap.shape[2]] = pointmap
                        pointmap = pad_pm
            return img, mask, pointmap

        # Prepend to img_mask_pointmap_joint_transform if it's empty or sentinel
        if (
            (self.img_mask_pointmap_joint_transform == (None,) or self.img_mask_pointmap_joint_transform is None)
            and (self.img_mask_joint_transform == (None,) or self.img_mask_joint_transform is None)
        ):
            # No joint transforms defined: set single crop/pad transform
            self.img_mask_pointmap_joint_transform = [ _crop_or_pad_joint ]
        elif self.img_mask_pointmap_joint_transform == (None,) or self.img_mask_pointmap_joint_transform is None:
            # There is a img_mask_joint_transform but no img_mask_pointmap_joint_transform, set crop/pad before existing list
            existing = self.img_mask_joint_transform if self.img_mask_joint_transform not in ((None,), None) else []
            self.img_mask_pointmap_joint_transform = [ _crop_or_pad_joint ] + list(existing)
        else:
            # Prepend crop/pad to the existing img_mask_pointmap_joint_transform list
            if isinstance(self.img_mask_pointmap_joint_transform, list):
                self.img_mask_pointmap_joint_transform = [ _crop_or_pad_joint ] + self.img_mask_pointmap_joint_transform
            else:
                # If it's a single callable, convert to list
                if self.img_mask_pointmap_joint_transform != (None,):
                    self.img_mask_pointmap_joint_transform = [ _crop_or_pad_joint, self.img_mask_pointmap_joint_transform ]


    def _normalize_pointmap(
        self, pointmap: torch.Tensor,
        mask: torch.Tensor,
        pointmap_normalizer: Callable,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ):
        if pointmap is None:
            return pointmap, None, None

        if self.normalize_pointmap == False:
            # old behavior: Pose is normalized to the pointmap center, but pointmap is not
            _, pointmap_scale, pointmap_shift = pointmap_normalizer.normalize(pointmap, mask)
            return pointmap, pointmap_scale, pointmap_shift
        
        if scale is not None or shift is not None:
            return pointmap_normalizer.normalize(pointmap, mask, scale, shift)
            
        return pointmap_normalizer.normalize(pointmap, mask)

    def _process_image_mask_pointmap_mess(
        self, rgb_image, rgb_image_mask, pointmap=None
    ):
        """Extended version that handles pointmaps"""
 
        # Apply pointmap normalization if enabled
        pointmap_for_crop, pointmap_scale, pointmap_shift = self._normalize_pointmap(
            pointmap, rgb_image_mask, self.pointmap_normalizer
        )

        # Apply transforms to the original full rgb image and mask.
        rgb_image, rgb_image_mask = self._preprocess_rgb_image_mask(rgb_image, rgb_image_mask)

        # These two are typically used for getting cropped images of the object
        #   : first apply joint transforms
        processed_rgb_image, processed_mask, processed_pointmap = (
            self._preprocess_image_mask_pointmap(rgb_image, rgb_image_mask, pointmap_for_crop)
        )
        #   : then apply individual transforms on top of the joint transforms
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, self.img_transform
        )
        processed_mask = self._apply_transform(processed_mask, self.mask_transform)
        if processed_pointmap is not None:
            processed_pointmap = self._apply_transform(
                processed_pointmap, self.pointmap_transform
            )

        # This version is typically the full version of the image
        #   : apply individual transforms only
        rgb_image = self._apply_transform(rgb_image, self.img_transform)
        rgb_image_mask = self._apply_transform(rgb_image_mask, self.mask_transform)
        
        rgb_pointmap, rgb_pointmap_scale, rgb_pointmap_shift = self._normalize_pointmap(
            pointmap, rgb_image_mask, self.rgb_pointmap_normalizer, pointmap_scale, pointmap_shift
        )

        if rgb_pointmap is not None:
            rgb_pointmap = self._apply_transform(rgb_pointmap, self.pointmap_transform)

        result = {
            "mask": processed_mask,
            "image": processed_rgb_image,
            "rgb_image": rgb_image,
            "rgb_image_mask": rgb_image_mask,
        }

        # Add pointmap results if available
        if processed_pointmap is not None:
            result.update(
                {
                    "pointmap": processed_pointmap,
                    "rgb_pointmap": rgb_pointmap,
                }
            )
            
        # Add normalization parameters if normalization was applied
        if pointmap_scale is not None and pointmap_shift is not None:
            result.update(
                {
                    "pointmap_scale": pointmap_scale,
                    "pointmap_shift": pointmap_shift,
                    "rgb_pointmap_scale": rgb_pointmap_scale,
                    "rgb_pointmap_shift": rgb_pointmap_shift,
                }
            )

        return result

    def _process_image_and_mask_mess(self, rgb_image, rgb_image_mask):
        """Original method - calls extended version without pointmap"""
        return self._process_image_mask_pointmap_mess(rgb_image, rgb_image_mask, None)

    def _preprocess_rgb_image_mask(self, rgb_image: torch.Tensor, rgb_image_mask: torch.Tensor):
        """Apply joint transforms to rgb_image and rgb_image_mask."""
        if (
            self.rgb_img_mask_joint_transform != (None,)
            and self.rgb_img_mask_joint_transform is not None
        ):
            for trans in self.rgb_img_mask_joint_transform:
                rgb_image, rgb_image_mask = trans(rgb_image, rgb_image_mask)
        return rgb_image, rgb_image_mask

    def _preprocess_image_mask_pointmap(self, rgb_image, mask_image, pointmap=None):
        """Apply joint transforms with priority: triple transforms > dual transforms."""
        # Priority: img_mask_pointmap_joint_transform when pointmap is provided
        if (
            self.img_mask_pointmap_joint_transform != (None,)
            and self.img_mask_pointmap_joint_transform is not None
            and pointmap is not None
        ):
            for trans in self.img_mask_pointmap_joint_transform:
                rgb_image, mask_image, pointmap = trans(
                    rgb_image, mask_image, pointmap=pointmap
                )
            return rgb_image, mask_image, pointmap

        # Fallback: img_mask_joint_transform (existing behavior)
        elif (
            self.img_mask_joint_transform != (None,)
            and self.img_mask_joint_transform is not None
        ):
            for trans in self.img_mask_joint_transform:
                rgb_image, mask_image = trans(rgb_image, mask_image)
            return rgb_image, mask_image, pointmap

        return rgb_image, mask_image, pointmap

    def _preprocess_image_and_mask(self, rgb_image, mask_image):
        """Backward compatibility wrapper - only applies dual transforms"""
        rgb_image, mask_image, _ = self._preprocess_image_mask_pointmap(
            rgb_image, mask_image, None
        )
        return rgb_image, mask_image

    # keep here for backward compatibility
    def _preprocess_image_and_mask_inference(self, rgb_image, mask_image):
        warnings.warn(
            "The _preprocess_image_and_mask_inference is deprecated! Please use _preprocess_image_and_mask",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._preprocess_image_and_mask(rgb_image, mask_image)

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None and transform is not None and transform != (None,):
            input = transform(input)

        return input