#!/usr/bin/env python3
"""Validate pipeline.yaml checkpoint references under a checkpoint folder.

Usage:
  python scripts/validate_pipeline_checkpoints.py checkpoints/<tag>/pipeline.yaml

This script loads the pipeline config and checks whether the referenced checkpoint files
exist and flags duplicate formats (.pt/.ckpt/.safetensors) or potential issues.
"""
import sys
from pathlib import Path
import yaml
import argparse
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from matplotlib.table import Table


def find_files_with_variants(base: Path, stem: str):
    # returns dict ext -> Path
    out = {}
    for ext in ['.safetensors', '.pt', '.ckpt', '.pth']:
        p = base / f"{stem}{ext}"
        if p.exists():
            out[ext] = p
    return out


def validate_pipeline_yaml(path: Path, output_dir: Path | None = None, visualize: bool = False):
    if not path.exists():
        print(f"Config not found: {path}")
        return 1
    base = path.parent

    with open(path) as f:
        config = yaml.safe_load(f)

    # list of expected keys for ckpt files used in InferencePipeline
    checkpoint_keys = [
        'ss_generator_ckpt_path',
        'slat_generator_ckpt_path',
        'ss_decoder_ckpt_path',
        'ss_encoder_ckpt_path',
        'slat_decoder_gs_ckpt_path',
        'slat_decoder_gs_4_ckpt_path',
        'slat_decoder_mesh_ckpt_path',
    ]

    issues = []
    reported = set()
    results = []

    for key in checkpoint_keys:
        # try case-insensitive variants
        val = config.get(key, None)
        if val is None:
            # sometimes config has different keys (without _path)
            val = config.get(key.replace('_ckpt_path', '_ckpt'), None)
        if val is None:
            continue
        # path may be relative inside the checkpoint folder or direct filename
        p = Path(val)
        p_rel = (base / p)
        if p_rel.exists():
            p = p_rel
        else:
            # check if any variant present in base
            stem = p.name.split('.')[0]
            variants = find_files_with_variants(base, stem)
            if variants:
                print(f"Found variants for {stem}: {', '.join(str(v) for v in variants.values())}")
            issues.append((key, val, str(p), p.exists(), variants if 'variants' in locals() else {}))
            results.append({
                'key': key,
                'value': val,
                'resolved': str(p),
                'exists': p.exists(),
                'variants': [str(v) for v in (variants.values() if 'variants' in locals() else [])],
            })

    # Extra scan: list duplicates for same stem
    cp_files = list(base.glob('*.*'))
    stems = {}
    for f in cp_files:
        if f.suffix in ['.pt', '.ckpt', '.safetensors', '.pth']:
            stems.setdefault(f.stem, []).append(f)

    duplicates = {k: v for k, v in stems.items() if len(v) > 1}
    if duplicates:
        print('Found multiple formats for these checkpoint stems (possible duplicates):')
        for s, fs in duplicates.items():
            print(f"  {s}: {', '.join(str(x.name) for x in fs)}")
            # record duplicate stems
            for f in fs:
                results.append({
                    'key': 'duplicate_stem',
                    'value': s,
                    'resolved': str(f),
                    'exists': True,
                    'variants': [str(x) for x in fs],
                })

    # Report issues
    if issues:
        print('\nPipeline YAML reference issues found:')
        for key, val, resolved, exists, variants in issues:
            print(f"Key: {key} -> {val}; Resolved: {resolved}; Exists: {exists}")
            if variants:
                print(f"    Variants found: {', '.join(str(x) for x in variants.values())}")
        if visualize and output_dir is not None:
            _create_visualization(results, duplicates, output_dir)
        return 1
    else:
        print('No missing checkpoint references detected by basic validation.')
        if duplicates:
            print('\nNote: multiple formats detected for the same stem; prefer `.safetensors` where possible for security and portability.')
        if visualize and output_dir is not None:
            _create_visualization(results, duplicates, output_dir)
        return 0


def _create_visualization(results, duplicates, output_dir: Path):
    """Create a simple PNG/HTML summary of validation results.

    - Bar chart: counts of exists/missing/variants/duplicates
    - Table with details for issues or duplicates (HTML)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stats
    counts = defaultdict(int)
    for r in results:
        if r.get('key') == 'duplicate_stem':
            counts['duplicates'] += 1
        else:
            counts['exists' if r.get('exists') else 'missing'] += 1
            if r.get('variants') and len(r.get('variants')) > 0:
                counts['variants'] += 1

    # Bar chart
    categories = ['exists', 'missing', 'variants', 'duplicates']
    values = [counts.get(c, 0) for c in categories]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, values, color=['#2ca02c', '#d62728', '#ff7f0e', '#9467bd'])
    ax.set_title('Pipeline Checkpoint Validation Summary')
    ax.set_ylabel('Count')
    plt.tight_layout()
    chart_path = output_dir / 'validation_summary.png'
    fig.savefig(chart_path)
    plt.close(fig)

    # Create HTML report with table
    rows = []
    for r in results:
        rows.append(r)

    html_lines = ["<html><head><meta charset='utf-8'><title>Pipeline Validation Report</title></head><body>"]
    html_lines.append('<h1>Pipeline Validation Report</h1>')
    html_lines.append(f"<img src='{chart_path.name}' alt='Summary' style='max-width:600px;'>")
    html_lines.append('<h2>Details</h2>')
    html_lines.append('<table border="1" cellpadding="4"><tr><th>key</th><th>value</th><th>resolved</th><th>exists</th><th>variants</th></tr>')
    for r in rows:
        html_lines.append(f"<tr><td>{r.get('key')}</td><td>{r.get('value')}</td><td>{r.get('resolved')}</td><td>{r.get('exists')}</td><td>{','.join(r.get('variants', []))}</td></tr>")
    html_lines.append('</table>')
    html_lines.append('</body></html>')
    report_path = output_dir / 'validation_report.html'
    with open(report_path, 'w') as f:
        f.write('\n'.join(html_lines))
    print(f"Visualization saved: {chart_path} and {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline', type=str, help='Path to pipeline.yaml')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for reports (defaults to pipeline parent dir /report)')
    parser.add_argument('--visualize', action='store_true', help='Create PNG/HTML summary report for the validation')
    args = parser.parse_args()
    # Determine output dir
    outdir = None
    if args.output_dir is None:
        outdir = Path(args.pipeline).parent / 'report'
    else:
        outdir = Path(args.output_dir)

    rc = validate_pipeline_yaml(Path(args.pipeline), output_dir=outdir, visualize=args.visualize)
    sys.exit(rc)


if __name__ == '__main__':
    main()
