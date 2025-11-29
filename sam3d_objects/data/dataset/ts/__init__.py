"""TS (TotalSegmentator) dataset loaders for SAM3D medical fine-tuning."""

# Lazy imports to avoid loading heavy dependencies
__all__ = [
    "TS_nnUNet_Dataset",
    "TS_SAM3D_Dataset",
    "data_collate",
    "SliceAugmentor",
    "create_augmentor",
]


def __getattr__(name):
    if name == "TS_nnUNet_Dataset":
        from sam3d_objects.data.dataset.ts.ts_nnunet_dataloader import TS_nnUNet_Dataset

        return TS_nnUNet_Dataset
    elif name == "TS_SAM3D_Dataset":
        from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import TS_SAM3D_Dataset

        return TS_SAM3D_Dataset
    elif name == "data_collate":
        from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import data_collate

        return data_collate
    elif name == "SliceAugmentor":
        from sam3d_objects.data.dataset.ts.slice_augmentations import SliceAugmentor

        return SliceAugmentor
    elif name == "create_augmentor":
        from sam3d_objects.data.dataset.ts.slice_augmentations import create_augmentor

        return create_augmentor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
