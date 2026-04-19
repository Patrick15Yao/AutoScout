#!/usr/bin/env python3
"""
Run four YOLO models (2 x detection, 2 x segmentation) on a single JPG and
write plain-text results. No visualization is produced.

Outputs:
  <out_dir>/<out_name>/txt/
    - players.txt     (YOLO det format:  class cx cy w h   normalized)
    - pocket.txt      (YOLO det format:  class cx cy w h   normalized)
    - yardline.txt    (YOLO seg format:  class x1 y1 x2 y2 ...  normalized)
    - harshmark.txt   (YOLO seg format:  class x1 y1 x2 y2 ...  normalized)

Notes:
- A "weight" argument may be either a direct .pt path or a run directory; in
  the latter case we resolve to <run>/weights/best.pt (falling back to last.pt).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import json
import torch

from ultralytics import YOLO
from mask2line import process_pair
from Visualize_1 import visualize_from_txt


DEFAULT_CONF = 0.25


@dataclass
class InferenceConfig:
	image_path: Path
	players_weights: Path
	pocket_weights: Path
	yardline_weights: Path
	harshmark_weights: Path
	conf_players: float = DEFAULT_CONF
	conf_pocket: float = DEFAULT_CONF
	conf_yardline: float = DEFAULT_CONF
	conf_harshmark: float = DEFAULT_CONF
	device: str = "auto"  # e.g., "cpu" or "0"
	out_dir: Path = Path("./out")
	out_name: str = "inference_result"


def _resolve_weight_path(p: Path) -> Path:
	"""
	Accept either a .pt file or a run directory. If directory, try weights/best.pt then weights/last.pt.
	"""
	p = p.expanduser().resolve()
	if p.is_file() and p.suffix == ".pt":
		return p
	if p.is_dir():
		best = p / "weights" / "best.pt"
		if best.exists():
			return best.resolve()
		last = p / "weights" / "last.pt"
		if last.exists():
			return last.resolve()
	raise FileNotFoundError(f"Cannot resolve weight file from: {p}")


def _ensure_out_dirs(base: Path, name: str) -> Path:
	"""
	Create output base directory and its 'txt' subdir. Return the base/<name> path.
	"""
	target = (base / name).resolve()
	(target / "txt").mkdir(parents=True, exist_ok=True)
	return target


def _write_detection_txt(result, out_path: Path) -> None:
	"""
	Write YOLO detection format lines: 'class cx cy w h' with normalized coords.
	"""
	lines: List[str] = []
	boxes = getattr(result, "boxes", None)
	if boxes is not None and getattr(boxes, "xywhn", None) is not None:
		xywhn = boxes.xywhn.tolist()
		classes = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else [0] * len(xywhn)
		for (cx, cy, w, h), c in zip(xywhn, classes):
			lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
	out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _write_detection_txt_with_conf(result, out_path: Path) -> None:
	"""
	Write YOLO detection format lines with confidence:
	'class cx cy w h conf' with normalized coords.
	"""
	lines: List[str] = []
	boxes = getattr(result, "boxes", None)
	if boxes is not None and getattr(boxes, "xywhn", None) is not None:
		xywhn = boxes.xywhn.tolist()
		classes = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else [0] * len(xywhn)
		confs = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xywhn)
		for (cx, cy, w, h), c, conf in zip(xywhn, classes, confs):
			lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(conf):.6f}")
	out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _write_segmentation_txt(result, out_path: Path) -> None:
	"""
	Write YOLO segmentation format lines: 'class x1 y1 x2 y2 ...' with normalized coords.
	All mask polygons are assumed class 0 (single-class models). If multi-class mask
	output is available, adapt here accordingly.
	"""
	lines: List[str] = []
	masks = getattr(result, "masks", None)
	if masks is not None and getattr(masks, "xy", None) is not None:
		h, w = result.orig_shape[:2]
		for poly in masks.xy:  # list of (N,2) arrays in pixel coords
			if poly is None or len(poly) == 0:
				continue
			coords = []
			for x, y in poly:
				xn = max(0.0, min(1.0, float(x) / float(w)))
				yn = max(0.0, min(1.0, float(y) / float(h)))
				coords.append(f"{xn:.6f} {yn:.6f}")
			lines.append("0 " + " ".join(coords))
	out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def infer_four_models(cfg: InferenceConfig) -> Path:
	"""
	Run four models (players det, pocket det, yardline seg, harshmark seg) on one image.
	Returns the output base directory (…/out_dir/out_name).
	"""
	# Resolve all weight paths.
	players_w = _resolve_weight_path(cfg.players_weights)
	pocket_w = _resolve_weight_path(cfg.pocket_weights)
	yard_w = _resolve_weight_path(cfg.yardline_weights)
	harsh_w = _resolve_weight_path(cfg.harshmark_weights)

	# Prepare output structure.
	out_base = _ensure_out_dirs(cfg.out_dir, cfg.out_name)
	txt_dir = out_base / "txt"

	# Players detection
	players_model = YOLO(str(players_w))
	res_p = players_model.predict(
	    source=str(cfg.image_path), device=cfg.device, conf=cfg.conf_players, verbose=False
	)[0]
	_write_detection_txt(res_p, txt_dir / "players.txt")
	_write_detection_txt_with_conf(res_p, txt_dir / "players_conf.txt")

	# Pocket/LOS detection
	pocket_model = YOLO(str(pocket_w))
	res_pk = pocket_model.predict(
	    source=str(cfg.image_path), device=cfg.device, conf=cfg.conf_pocket, verbose=False
	)[0]
	_write_detection_txt(res_pk, txt_dir / "pocket.txt")
	_write_detection_txt_with_conf(res_pk, txt_dir / "pocket_conf.txt")

	# YardLine segmentation
	yard_model = YOLO(str(yard_w))
	res_y = yard_model.predict(
	    source=str(cfg.image_path), device=cfg.device, conf=cfg.conf_yardline, verbose=False
	)[0]
	_write_segmentation_txt(res_y, txt_dir / "yardline.txt")

	# HarshMark segmentation
	harsh_model = YOLO(str(harsh_w))
	res_h = harsh_model.predict(
	    source=str(cfg.image_path), device=cfg.device, conf=cfg.conf_harshmark, verbose=False
	)[0]
	_write_segmentation_txt(res_h, txt_dir / "harshmark.txt")

	print(f"[done] Wrote TXT outputs to: {txt_dir}")
	return out_base


def _add_common_args(ap: argparse.ArgumentParser) -> None:
	ap.add_argument("--image", required=True, help="Path to a single JPG image to infer.")
	ap.add_argument("--config", help="Optional JSON config file specifying weights, thresholds, device, and out_name.")
	ap.add_argument("--players-weights", help="Players detector weights (.pt) or run dir. Required unless provided in --config.")
	ap.add_argument("--pocket-weights", help="Pocket/LOS detector weights (.pt) or run dir. Required unless provided in --config.")
	ap.add_argument("--yardline-weights", help="YardLine segmentation weights (.pt) or run dir. Required unless provided in --config.")
	ap.add_argument("--harshmark-weights", help="HarshMark segmentation weights (.pt) or run dir. Required unless provided in --config.")
	# Use None defaults so JSON can override; we apply real defaults after merge
	ap.add_argument("--conf-players", type=float, default=None, help=f"Confidence for players detector (default {DEFAULT_CONF}).")
	ap.add_argument("--conf-pocket", type=float, default=None, help=f"Confidence for pocket/LOS detector (default {DEFAULT_CONF}).")
	ap.add_argument("--conf-yardline", type=float, default=None, help=f"Confidence for yardline segmenter (default {DEFAULT_CONF}).")
	ap.add_argument("--conf-harshmark", type=float, default=None, help=f"Confidence for harshmark segmenter (default {DEFAULT_CONF}).")
	ap.add_argument("--device", default=None, help="Inference device, e.g., 'cpu' or '0'. Use 'auto' to pick GPU if available else CPU.")
	ap.add_argument("--out-dir", required=True, help="Base output directory.")
	ap.add_argument("--out-name", help="Name of subdirectory to create under out-dir. If omitted, can be provided in config or defaults to 'inference_result'.")
	ap.add_argument("--visualize", action="store_true", help="If set, render visualization JPGs into <out-dir>/<out-name>/vis using generated TXT files.")


def _load_config_json(path_str: Optional[str]) -> dict:
	if not path_str:
		return {}
	path = Path(path_str).expanduser().resolve()
	if not path.exists():
		raise FileNotFoundError(f"Config JSON not found: {path}")
	with path.open("r") as f:
		return json.load(f)


def _effective_value(cli_value, cfg: dict, key: str, default=None):
	"""
	Priority: CLI explicit (non-None/non-empty) > config JSON key > default
	"""
	if cli_value is not None and cli_value != "":
		return cli_value
	if key in cfg and cfg[key] not in (None, ""):
		return cfg[key]
	return default


def _resolve_device_string(device_str: str) -> str:
	"""
	Map 'auto' to '0' if CUDA is available, else 'cpu'. Pass-through otherwise.
	"""
	if isinstance(device_str, str) and device_str.lower() == "auto":
		return "0" if torch.cuda.is_available() else "cpu"
	return device_str


def main() -> None:
	ap = argparse.ArgumentParser(description="Run four YOLO models (2x det, 2x seg) on a single JPG and write TXT outputs.")
	_add_common_args(ap)
	args = ap.parse_args()

	cfg_json = _load_config_json(getattr(args, "config", None))

	# Default out_name falls back to image filename stem if not provided in CLI/JSON
	default_out_name = Path(args.image).stem if getattr(args, "image", None) else "inference_result"

	# Resolve values with precedence CLI > JSON > default
	players_weights_val = _effective_value(args.players_weights, cfg_json, "players_weights")
	pocket_weights_val = _effective_value(args.pocket_weights, cfg_json, "pocket_weights")
	yardline_weights_val = _effective_value(args.yardline_weights, cfg_json, "yardline_weights")
	harshmark_weights_val = _effective_value(args.harshmark_weights, cfg_json, "harshmark_weights")
	if not players_weights_val or not pocket_weights_val or not yardline_weights_val or not harshmark_weights_val:
		ap.error("Missing weights. Provide via CLI flags or --config JSON (players_weights, pocket_weights, yardline_weights, harshmark_weights).")

	cfg = InferenceConfig(
	    image_path=Path(args.image),
	    players_weights=Path(players_weights_val),
	    pocket_weights=Path(pocket_weights_val),
	    yardline_weights=Path(yardline_weights_val),
	    harshmark_weights=Path(harshmark_weights_val),
	    conf_players=float(_effective_value(args.conf_players, cfg_json, "conf_players", DEFAULT_CONF)),
	    conf_pocket=float(_effective_value(args.conf_pocket, cfg_json, "conf_pocket", DEFAULT_CONF)),
	    conf_yardline=float(_effective_value(args.conf_yardline, cfg_json, "conf_yardline", DEFAULT_CONF)),
	    conf_harshmark=float(_effective_value(args.conf_harshmark, cfg_json, "conf_harshmark", DEFAULT_CONF)),
	    device=str(_effective_value(args.device, cfg_json, "device", "auto")),
	    out_dir=Path(args.out_dir),
	    out_name=str(_effective_value(args.out_name, cfg_json, "out_name", default_out_name)),
	)

	# Resolve 'auto' device here to avoid Ultralytics rejecting it
	cfg.device = _resolve_device_string(cfg.device)
	out_base = infer_four_models(cfg)
	txt_dir = out_base / "txt"
	# Post-process segmentation masks into fitted line segments (yardline/harshmark).
	process_pair(txt_dir, txt_dir)
	# Optionally visualize outputs next to txt/ as vis/.
	if args.visualize:
		visualize_from_txt(
		    image_path=cfg.image_path,
		    txt_dir=txt_dir,
		    out_dir=out_base,
		    line_width=2,
		    mask_alpha=0.35,
		)


if __name__ == "__main__":
	main()


