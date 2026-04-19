#!/usr/bin/env python3
"""
Run inference_all.py then Color_Clustering.py using only image path + output path.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Default weight locations (edit here if your run folders change)
PLAYERS_WEIGHTS = ROOT / "runs" / "players_training_2_cont2"
POCKET_WEIGHTS = ROOT / "runs" / "LOS_Training_1_"
YARDLINE_WEIGHTS = ROOT / "runs" / "yardline_Training_3_"
HARSHMARK_WEIGHTS = ROOT / "runs" / "HarshMark_Training_1_"

INFER_SCRIPT = ROOT / "code" / "Inference_tgt" / "inference_all.py"
CLUSTER_SCRIPT = ROOT / "code" / "ColorLab" / "Color_Clustering.py"


def run(cmd: list[str]) -> None:
	print("[run]", " ".join(str(c) for c in cmd))
	subprocess.run(cmd, check=True)


def main() -> None:
	ap = argparse.ArgumentParser(description="Run detection + color clustering with only image and output paths.")
	ap.add_argument("--image", required=True, help="Path to input image.")
	ap.add_argument("--out-dir", required=True, help="Output directory for all results.")
	args = ap.parse_args()

	image_path = Path(args.image).resolve()
	out_dir = Path(args.out_dir).resolve()
	out_name = image_path.stem

	run([
		"python",
		str(INFER_SCRIPT),
		"--image",
		str(image_path),
		"--players-weights",
		str(PLAYERS_WEIGHTS),
		"--pocket-weights",
		str(POCKET_WEIGHTS),
		"--yardline-weights",
		str(YARDLINE_WEIGHTS),
		"--harshmark-weights",
		str(HARSHMARK_WEIGHTS),
		"--out-dir",
		str(out_dir),
		"--out-name",
		out_name,
		"--visualize",
	])

	run([
		"python",
		str(CLUSTER_SCRIPT),
		"--image",
		str(image_path),
		"--boxes",
		str(out_dir / out_name / "txt" / "players.txt"),
		"--out-dir",
		str(out_dir / out_name / "Indi"),
	])


if __name__ == "__main__":
	main()
