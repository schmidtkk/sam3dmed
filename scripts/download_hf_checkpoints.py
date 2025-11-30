#!/usr/bin/env python3
"""Download checkpoint files from a Hugging Face repo into `checkpoints/hf/`.

Usage:
  python scripts/download_hf_checkpoints.py --repo facebook/sam-3d-objects --out checkpoints/hf

Notes:
- If the HF repo is gated (private/allowlist) you will need to authenticate using a token with
  access, e.g. `export HF_TOKEN=...` or `huggingface-cli login` before running.
"""
import argparse
import os
import sys
from pathlib import Path
import logging

try:
    from huggingface_hub import HfApi, hf_hub_download
except Exception:
    print("Please install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


def download_repo_files(repo_id: str, out_dir: Path, pattern_exts=None):
    api = HfApi()

    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        logging.error("Failed to list repo files for %s: %s", repo_id, e)
        if 'hf_hub' in str(e).lower() or '403' in str(e):
            logging.error("You likely need to authenticate or request access to the gated HF repo.")
        raise

    if pattern_exts is None:
        pattern_exts = ['.pt', '.ckpt', '.safetensors', '.pth']

    # Filter by ext
    to_download = [f for f in files if Path(f).suffix in pattern_exts]
    if not to_download:
        logging.info("No checkpoint files found with extensions %s in repo %s", pattern_exts, repo_id)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %d files from %s -> %s", len(to_download), repo_id, out_dir)

    for f in to_download:
        try:
            logging.info("Downloading %s ...", f)
            downloaded = hf_hub_download(repo_id, filename=f, repo_type='model')
            # hf_hub_download returns a local file path, copy to out_dir preserving filename
            dest = out_dir / Path(f).name
            if Path(downloaded) != dest:
                import shutil
                shutil.copy(downloaded, dest)
            # Validate: ensure the copied file is non-empty
            try:
                size = dest.stat().st_size
                if size == 0:
                    logging.warning("Downloaded file %s is zero bytes. It may be a stub or gated file.", dest)
            except Exception:
                logging.warning("Could not stat file %s after copying", dest)
            logging.info("Saved: %s", dest)
        except Exception as e:
            logging.warning("Failed to download %s: %s", f, e)
            # Don't abort; continue with others


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=str, default='facebook/sam-3d-objects')
    parser.add_argument('--out', type=str, default='checkpoints/hf')
    parser.add_argument('--exts', type=str, default='.pt,.ckpt,.safetensors,.pth')
    args = parser.parse_args()

    out = Path(args.out)
    repo = args.repo
    exts = [e.strip() for e in args.exts.split(',') if e.strip()]

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting HF download for repo %s", repo)

    try:
        download_repo_files(repo, out, exts)
    except Exception as e:
        logging.error("Could not download files: %s", e)
        logging.error("Make sure you have a token with access to the repo, e.g. export HF_TOKEN=<token>")
        logging.error("You can authenticate using `huggingface-cli login` or set HF_TOKEN in the environment.")
        sys.exit(1)

    logging.info("HF download completed (best-effort). Check %s for files.", out)


if __name__ == '__main__':
    main()
