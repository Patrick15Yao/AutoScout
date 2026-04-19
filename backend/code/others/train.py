import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO


DEFAULT_DATA_ROOT = "/userhome/cs/u3597462/Autoscout/Data/Structured_Dataset"
DEFAULT_PROJECT_DIR = "/userhome/cs/u3597462/Autoscout/runs"
DEFAULT_MODEL = "yolov8n.pt"


def find_data_yaml(data_arg: str) -> str:
	"""
	Resolve a dataset YAML path.
	- If a YAML file path is provided, return it.
	- If a directory is provided, prefer 'data.yaml' inside it; otherwise pick the first *.yaml/*.yml.
	"""
	path = Path(data_arg).expanduser().resolve()
	if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
		return str(path)

	if path.is_dir():
		candidate = path / "data.yaml"
		if candidate.exists():
			return str(candidate)
		yamls = list(path.glob("*.yml")) + list(path.glob("*.yaml"))
		if yamls:
			return str(sorted(yamls)[0])

	raise FileNotFoundError(
		f"Could not find dataset YAML. Provide a file or a directory containing a YAML. Got: {data_arg}"
	)


def find_latest_checkpoint(project_dir: str) -> Optional[str]:
	"""
	Find the most recently modified YOLO last checkpoint under project_dir.
	Search pattern '**/weights/last.pt'.
	"""
	root = Path(project_dir).expanduser().resolve()
	if not root.exists():
		return None
	checkpoints = list(root.rglob("weights/last.pt"))
	if not checkpoints:
		return None
	checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	return str(checkpoints[0])


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train/Resume YOLOv8 conveniently.")
	parser.add_argument(
		"--data",
		type=str,
		default=DEFAULT_DATA_ROOT,
		help="Path to dataset YAML or directory containing it.",
	)
	parser.add_argument(
		"--start",
		type=str,
		choices=["auto", "scratch", "resume"],
		default="auto",
		help="Training start mode: 'scratch' uses the pretrained model; 'resume' continues from latest checkpoint; 'auto' prefers resume if available.",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=50,
		help="Number of training epochs.",
	)
	parser.add_argument(
		"--imgsz",
		type=int,
		default=640,
		help="Image size.",
	)
	parser.add_argument(
		"--batch",
		type=int,
		default=16,
		help="Batch size.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=DEFAULT_MODEL,
		help="Pretrained model weights to start from when --start=scratch, e.g. yolov8n.pt/yolov8s.pt/....",
	)
	parser.add_argument(
		"--project",
		type=str,
		default=DEFAULT_PROJECT_DIR,
		help="Runs/project directory.",
	)
	parser.add_argument(
		"--name",
		type=str,
		default="yolov8-custom",
		help="Run name inside the project directory.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		help="Device to use, e.g., '0' for GPU 0, 'cpu' for CPU, or 'auto' to pick automatically.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Random seed.",
	)
	return parser.parse_args()


def resolve_device_arg(device: str) -> str:
	if device == "auto":
		return "0" if torch.cuda.is_available() else "cpu"
	return device


def main() -> None:
	args = parse_args()

	data_yaml = find_data_yaml(args.data)
	project_dir = Path(args.project).expanduser().resolve()
	project_dir.mkdir(parents=True, exist_ok=True)

	device = resolve_device_arg(args.device)
	print(f"[info] torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | device={device}")

	last_ckpt = find_latest_checkpoint(str(project_dir))
	print(f"[info] latest checkpoint under '{project_dir}': {last_ckpt or 'None'}")

	resume_flag = False
	model_path = args.model

	if args.start == "resume":
		if last_ckpt and Path(last_ckpt).exists():
			model_path = last_ckpt
			resume_flag = True
			print(f"[info] Resuming from: {model_path}")
		else:
			print("[warn] --start=resume requested but no checkpoint found. Falling back to scratch.")
	elif args.start == "auto":
		if last_ckpt and Path(last_ckpt).exists():
			model_path = last_ckpt
			resume_flag = True
			print(f"[info] Auto mode: resuming from: {model_path}")
		else:
			print("[info] Auto mode: no checkpoint found, starting from scratch.")
	else:
		print(f"[info] Starting from scratch with pretrained model: {model_path}")

	model = YOLO(model_path)

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
	)

	print("\n[done] Training finished.")
	print(f"[done] Results dir: {getattr(results, 'save_dir', 'see runs/ directory')}")
	# Try to display the path to the new 'last.pt'
	new_last = find_latest_checkpoint(str(project_dir))
	if new_last:
		print(f"[done] Latest checkpoint: {new_last}")
	else:
		print("[done] No 'last.pt' found yet (this may appear after first epoch).")


if __name__ == "__main__":
	main()