"""Build a checked, optional RootSIFT gallery from reconciled references."""
import argparse
from pathlib import Path
from app.catalog import load_catalog
from app.local_features import LocalFeatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--references', required=True, type=Path)
    parser.add_argument('--images-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    engine = LocalFeatures()
    engine.build(args.references, args.images_dir, args.output, load_catalog(args.catalog))


if __name__ == '__main__':
    main()
