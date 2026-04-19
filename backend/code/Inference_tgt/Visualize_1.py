#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _ensure_vis_dir(out_dir: Path) -> Path:
	vis_dir = out_dir / "vis"
	vis_dir.mkdir(parents=True, exist_ok=True)
	return vis_dir


def _load_detection_txt(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
	"""Parse YOLO det format lines: class cx cy w h (normalized)."""
	items: List[Tuple[int, float, float, float, float]] = []
	if not txt_path.exists():
		return items
	with txt_path.open("r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split()
			if len(parts) < 5:
				continue
			try:
				c = int(float(parts[0]))
				cx = float(parts[1])
				cy = float(parts[2])
				w = float(parts[3])
				h = float(parts[4])
				items.append((c, cx, cy, w, h))
			except Exception:
				continue
	return items


def _load_segmentation_txt(txt_path: Path) -> List[List[Tuple[float, float]]]:
	"""Parse YOLO seg format lines: class x1 y1 x2 y2 ... (normalized)."""
	polys: List[List[Tuple[float, float]]] = []
	if not txt_path.exists():
		return polys
	with txt_path.open("r") as f:
		for line in f:
			# strip whitespace; skip empty lines
			line = line.strip()
			if not line:
				continue
			parts = line.split()
			if len(parts) < 3:
				continue
			try:
				# first value is class id; skip it
				coords = list(map(float, parts[1:]))
			except Exception:
				continue
			if len(coords) < 4 or len(coords) % 2 != 0:
				continue
			pairs: List[Tuple[float, float]] = []
			for i in range(0, len(coords), 2):
				pairs.append((coords[i], coords[i + 1]))
			polys.append(pairs)
	return polys


def _load_line_segments_txt(txt_path: Path) -> List[Tuple[float, float, float, float]]:
	"""Parse line segments file where each line is: x1 y1 x2 y2 (normalized)."""
	segs: List[Tuple[float, float, float, float]] = []
	if not txt_path.exists():
		return segs
	with txt_path.open("r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split()
			if len(parts) < 4:
				continue
			try:
				x1 = float(parts[0]); y1 = float(parts[1]); x2 = float(parts[2]); y2 = float(parts[3])
			except Exception:
				continue
			segs.append((x1, y1, x2, y2))
	return segs


def _draw_white_boxes(base_img: np.ndarray, dets: List[Tuple[int, float, float, float, float]], line_w: int = 2) -> np.ndarray:
	"""Draw white rectangles (no labels) on a copy of base_img. dets in normalized cx,cy,w,h."""
	h, w = base_img.shape[:2]
	out = base_img.copy()
	for _, cx, cy, bw, bh in dets:
		x1 = int((cx - bw / 2.0) * w)
		y1 = int((cy - bh / 2.0) * h)
		x2 = int((cx + bw / 2.0) * w)
		y2 = int((cy + bh / 2.0) * h)
		x1 = max(0, min(w - 1, x1))
		y1 = max(0, min(h - 1, y1))
		x2 = max(0, min(w - 1, x2))
		y2 = max(0, min(h - 1, y2))
		cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=line_w)
	return out


def _draw_white_masks(base_img: np.ndarray, polys: List[List[Tuple[float, float]]], alpha: float = 0.35, line_w: int = 2) -> np.ndarray:
	"""Draw filled white masks (semi-transparent) and white borders for each polygon defined in normalized coords."""
	h, w = base_img.shape[:2]
	out = base_img.copy()
	overlay = out.copy()
	for poly in polys:
		if len(poly) < 2:
			continue
		pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)
		pts = pts.reshape((-1, 1, 2))
		# fill
		cv2.fillPoly(overlay, [pts], color=(255, 255, 255))
		# border
		cv2.polylines(out, [pts], isClosed=True, color=(255, 255, 255), thickness=line_w)
	# blend
	cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
	return out


def _draw_red_lines(base_img: np.ndarray, segs: List[Tuple[float, float, float, float]], line_w: int = 2) -> np.ndarray:
	"""Draw red straight line segments defined in normalized coords onto a copy of base_img."""
	h, w = base_img.shape[:2]
	out = base_img.copy()
	for x1, y1, x2, y2 in segs:
		p1 = (int(x1 * w), int(y1 * h))
		p2 = (int(x2 * w), int(y2 * h))
		cv2.line(out, p1, p2, (0, 0, 255), thickness=line_w)
	return out


def visualize_from_txt(
	image_path: Path,
	txt_dir: Path,
	out_dir: Path,
	line_width: int = 2,
	mask_alpha: float = 0.35,
) -> Path:
	"""
	Read four txt files (players.txt, pocket.txt, yardline.txt, harshmark.txt) from txt_dir,
	draw on image, and write four JPGs into <out_dir>/vis/.
	Returns the vis directory path.
	"""
	img_path = image_path.expanduser().resolve()
	txt_dir = txt_dir.expanduser().resolve()
	out_dir = out_dir.expanduser().resolve()
	vis_dir = _ensure_vis_dir(out_dir)

	if not img_path.exists():
		raise FileNotFoundError(f"Image not found: {img_path}")
	if not txt_dir.exists():
		raise FileNotFoundError(f"TXT dir not found: {txt_dir}")

	base = cv2.imread(str(img_path))
	if base is None:
		raise RuntimeError(f"Failed to read image: {img_path}")
	stem = img_path.stem

	# Players / Pocket (detections)
	players_txt = txt_dir / "players.txt"
	pocket_txt = txt_dir / "pocket.txt"
	yardline_txt = txt_dir / "yardline.txt"
	harshmark_txt = txt_dir / "harshmark.txt"
	yardline_line_txt = txt_dir / "yardline_line.txt"
	harshmark_line_txt = txt_dir / "harshmark_line.txt"

	players_dets = _load_detection_txt(players_txt)
	pocket_dets = _load_detection_txt(pocket_txt)
	yardline_polys = _load_segmentation_txt(yardline_txt)
	harshmark_polys = _load_segmentation_txt(harshmark_txt)
	yardline_lines = _load_line_segments_txt(yardline_line_txt)
	harshmark_lines = _load_line_segments_txt(harshmark_line_txt)

	# Draw and save
	if players_dets:
		img_p = _draw_white_boxes(base, players_dets, line_w=line_width)
		cv2.imwrite(str(vis_dir / f"{stem}_players.jpg"), img_p)
	if pocket_dets:
		img_pk = _draw_white_boxes(base, pocket_dets, line_w=line_width)
		cv2.imwrite(str(vis_dir / f"{stem}_pocket.jpg"), img_pk)
	# Yardline: draw masks (if any), then overlay red fitted lines (if any)
	if yardline_polys or yardline_lines:
		canvas_y = base.copy()
		if yardline_polys:
			canvas_y = _draw_white_masks(canvas_y, yardline_polys, alpha=mask_alpha, line_w=line_width)
		if yardline_lines:
			canvas_y = _draw_red_lines(canvas_y, yardline_lines, line_w=line_width)
		cv2.imwrite(str(vis_dir / f"{stem}_yardline.jpg"), canvas_y)

	# Harshmark: draw masks (if any), then overlay red fitted lines (if any)
	if harshmark_polys or harshmark_lines:
		canvas_h = base.copy()
		if harshmark_polys:
			canvas_h = _draw_white_masks(canvas_h, harshmark_polys, alpha=mask_alpha, line_w=line_width)
		if harshmark_lines:
			canvas_h = _draw_red_lines(canvas_h, harshmark_lines, line_w=line_width)
		cv2.imwrite(str(vis_dir / f"{stem}_harshmark.jpg"), canvas_h)

	print(f"[done] Visuals written to: {vis_dir}")
	return vis_dir


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize 4 TXT annotations (players/pocket yardline/harshmark) onto a JPG.")
	parser.add_argument("--txt-dir", required=True, help="Directory containing players.txt, pocket.txt, yardline.txt, harshmark.txt")
	parser.add_argument("--image", required=True, help="Path to the JPG image to visualize.")
	parser.add_argument("--out-dir", required=True, help="Output base directory; 'vis' subfolder will be created here.")
	parser.add_argument("--line-width", type=int, default=2, help="Line width for boxes and mask borders.")
	parser.add_argument("--mask-alpha", type=float, default=0.35, help="Alpha for filled mask overlay (0..1).")
	args = parser.parse_args()

	visualize_from_txt(
	    image_path=Path(args.image),
	    txt_dir=Path(args.txt_dir),
	    out_dir=Path(args.out_dir),
	    line_width=args.line_width,
	    mask_alpha=args.mask_alpha,
	)


if __name__ == "__main__":
	main()


