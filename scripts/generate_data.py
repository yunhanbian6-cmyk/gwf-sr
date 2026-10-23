"""Generate the configured synthetic dataset for the current stage-two experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gwf_sr.config import load_config
from gwf_sr.data import generate_configured_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml", help="YAML experiment configuration.")
    parser.add_argument("--root", default=".", help="Root directory for generated data and results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = generate_configured_dataset(config, root=args.root)
    output_dir = Path(args.root) / config.output_dir / config.config_name / "data"
    print(f"Generated {len(records)} image sets in {output_dir}")
    for record in records:
        print(f"- {record.name}: HR={record.hr_size}, LR={record.lr_size}")


if __name__ == "__main__":
    main()
