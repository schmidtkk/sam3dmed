#!/usr/bin/env python3
"""Hydra-based wrapper for training using YAML config.

Usage:
  python scripts/train_medical_hydra.py --config-path configs --config-name train
  # OR pass overrides e.g:
  python scripts/train_medical_hydra.py training.batch_size=8 data.data_root=/path/to/preprocessed
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Allow imports from repo root
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

import os

os.environ.setdefault('LIDRA_SKIP_INIT', '1')
from scripts.train_medical import train_from_config


@hydra.main(version_base="1.1", config_path=str(root / 'configs'), config_name='train')
def main(cfg: DictConfig) -> None:
    print("Hydra config loaded:\n", OmegaConf.to_yaml(cfg))
    train_from_config(OmegaConf.to_container(cfg, resolve=True))


if __name__ == '__main__':
    main()
