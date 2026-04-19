#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO


# Defaults for your environment
DEFAULT_DATA_ROOT = "/userhome/cs/u3597462/Autoscout/Data/YardLine/Training1"
DEFAULT_OUT_DIR = "/userhome/cs/u3597462/Autoscout/runs"
DEFAULT_RUN_NAME = "yardline-seg"
DEFAULT_MODEL = "yolov8n-seg.pt"


def find_data_yaml(data_arg: str) -> str:
	"""
	Resolve a dataset YAML path for YOLO.
	- If a YAML file path is provided, return it.
	- If a directory is provided, return '<dir>/dataset.yaml' if present, otherwise the first *.yaml/yml.
	"""
	path = Path(data_arg).expanduser().resolve()
	if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
		return str(path)

	if path.is_dir():
		candidate = path / "dataset.yaml"
		if candidate.exists():
			return str(candidate)
		alt = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
		if alt:
			return str(sorted(alt)[0])

	raise FileNotFoundError(f"Could not find dataset YAML in: {data_arg}")


def find_latest_checkpoint(project_dir: str) -> Optional[str]:
	"""
	Find the newest 'weights/last.pt' under the given project directory.
	"""
	root = Path(project_dir).expanduser().resolve()
	if not root.exists():
		return None
	candidates = sorted(root.rglob("weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
	return str(candidates[0]) if candidates else None


def resolve_device(device_arg: str) -> str:
	if device_arg == "auto":
		return "0" if torch.cuda.is_available() else "cpu"
	return device_arg


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation (YardLine).")
	parser.add_argument("--data", type=str, default=DEFAULT_DATA_ROOT, help="Dataset root directory or YAML path.")
	parser.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="Output project directory for runs.")
	parser.add_argument("--name", type=str, default=DEFAULT_RUN_NAME, help="Run name inside the project directory.")
	parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
	parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
	parser.add_argument("--batch", type=int, default=16, help="Batch size.")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Starting model weights, e.g. yolov8n-seg.pt or a custom .pt.")
	parser.add_argument("--weights", type=str, default="", help="Optional previous checkpoint (.pt) to start from.")
	parser.add_argument("--resume", action="store_true", help="Resume from the latest 'weights/last.pt' found under --out.")
	parser.add_argument("--device", type=str, default="auto", help="Device: '0'.. for GPU, 'cpu', or 'auto'.")
	parser.add_argument("--seed", type=int, default=0, help="Random seed.")

	# Common photographic augmentations (can be tuned as needed)
	parser.add_argument("--degrees", type=float, default=10.0)
	parser.add_argument("--translate", type=float, default=0.1)
	parser.add_argument("--scale", type=float, default=0.5)
	parser.add_argument("--shear", type=float, default=0.0)
	parser.add_argument("--perspective", type=float, default=0.0)
	parser.add_argument("--fliplr", type=float, default=0.5)
	parser.add_argument("--hsv_h", type=float, default=0.015)
	parser.add_argument("--hsv_s", type=float, default=0.7)
	parser.add_argument("--hsv_v", type=float, default=0.4)
	parser.add_argument("--mosaic", type=float, default=1.0)
	parser.add_argument("--mixup", type=float, default=0.1)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	data_yaml = find_data_yaml(args.data)
	project_dir = Path(args.out).expanduser().resolve()
	project_dir.mkdir(parents=True, exist_ok=True)

	device = resolve_device(args.device)
	print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
	print(f"[info] data_yaml={data_yaml}")
	print(f"[info] project={project_dir} name={args.name}")

	# Determine starting weights and resume behavior
	resume_flag = False
	start_weights = args.model
	if args.resume:
		last_ckpt = find_latest_checkpoint(str(project_dir))
		if last_ckpt:
			resume_flag = True
			start_weights = last_ckpt
			print(f"[info] Resuming from: {start_weights}")
		else:
			print("[warn] --resume set but no checkpoint found under project; starting new run instead.")
	elif args.weights:
		start_weights = str(Path(args.weights).expanduser().resolve())
		print(f"[info] Starting from provided weights: {start_weights}")
	else:
		print(f"[info] Starting from base model: {start_weights}")

	model = YOLO(start_weights)

	results = model.train(
		data=data_yaml,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		project=str(project_dir),
		name=args.name,
		resume=resume_flag,
		device=device,
		seed=args.seed,
		# Photographic augmentations
		degrees=args.degrees,
		translate=args.translate,
		scale=args.scale,
		shear=args.shear,
		perspective=args.perspective,
		fliplr=args.fliplr,
		hsv_h=args.hsv_h,
		hsv_s=args.hsv_s,
		hsv_v=args.hsv_v,
		mosaic=args.mosaic,
		mixup=args.mixup,
	)

	print("\n[done] Training finished.")
	print(f"[done] Results dir: {getattr(results, 'save_dir', 'see runs/ directory')}")


if __name__ == "__main__":
	main()

