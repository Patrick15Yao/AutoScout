#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Set


IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Randomly sample N JPG images from a directory and copy them to an output directory."
	)
	parser.add_argument("--source", type=str, required=True, help="Input images directory (jpg/jpeg).")
	parser.add_argument("--out-dir", type=str, required=True, help="Output directory to copy sampled images into.")
	parser.add_argument("--num", type=int, required=True, help="Number of images to sample.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
	return parser.parse_args()


def list_images(directory: Path) -> List[Path]:
	return sorted([p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()])


def main() -> None:
	args = parse_args()

	src = Path(args.source).expanduser().resolve()
	if not src.exists():
		raise SystemExit(f"[error] Source directory not found: {src}")

	dst = Path(args.out_dir).expanduser().resolve()
	dst.mkdir(parents=True, exist_ok=True)

	all_imgs = list_images(src)
	if not all_imgs:
		raise SystemExit(f"[error] No JPG images found under: {src}")

	sample_n = max(1, min(args.num, len(all_imgs)))
	random.seed(args.seed)
	sampled = random.sample(all_imgs, sample_n)

	for p in sampled:
		shutil.copy2(p, dst / p.name)

	print(f"[done] Copied {len(sampled)} images to: {dst}")


if __name__ == "__main__":
	main()

