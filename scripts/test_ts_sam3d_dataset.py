#!/usr/bin/env python3
"""Quick test for TS_SAM3D_Dataset"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import TS_SAM3D_Dataset, data_collate

parser = argparse.ArgumentParser()
parser.add_argument("--nifti_dir", required=True)
parser.add_argument("--cache_dir", default=None)
args = parser.parse_args()

print("Loading dataset...")
ds = TS_SAM3D_Dataset(
    original_nifti_dir=args.nifti_dir,
    cache_slices=True,
    slice_cache_dir=args.cache_dir or Path(args.nifti_dir) / "slice_cache",
    classes=5,
)
loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=data_collate, num_workers=0)

for batch in loader:
    print("Batch keys:", list(batch.keys()))
    print("Image shape:", batch["image"].shape)
    print("Affine shape:", batch["affine"].shape)
    print("Names:", batch["name"])
    break

print("Done")
