#!/usr/bin/env python3
"""
Crop YOLO-format bounding boxes, compute LAB features, cluster into 2 teams,
and copy crops into cluster_0/ and cluster_1/ under the output directory.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_BOX_INDEX_RE = re.compile(r"^box_(\d+)_")


def _extract_box_index(path: Path) -> int | None:
	match = _BOX_INDEX_RE.match(path.name)
	if not match:
		return None
	return int(match.group(1))


def _write_team_assignments_json(
    assignments: Dict[int, int],
    cluster_feature_means: Dict[int, List[float]],
    out_parent: Path,
) -> Path:
	"""
	Write team cluster assignments indexed by players.txt row index and cluster feature means.
	"""
	out_path = out_parent / "team_assignments.json"
	payload = {
	    "format_version": 1,
	    "description": "player index (players.txt row index) -> team cluster (0 or 1)",
	    "assignments": [{"player_index": idx, "team_cluster": assignments[idx]} for idx in sorted(assignments.keys())],
	    "cluster_feature_means": [
	        {"team_cluster": c, "mean_feature": cluster_feature_means[c]} for c in sorted(cluster_feature_means.keys())
	    ],
	}
	out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
	print(f"[done] Wrote team assignments JSON: {out_path}")
	return out_path


def _read_boxes(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
	"""
	Read YOLO detection boxes: class cx cy w h (normalized).
	Returns list of (class_id, cx, cy, w, h).
	"""
	lines = txt_path.read_text().strip().splitlines() if txt_path.exists() else []
	boxes: List[Tuple[int, float, float, float, float]] = []
	for line in lines:
		if not line.strip():
			continue
		parts = line.strip().split()
		if len(parts) < 5:
			continue
		try:
			class_id = int(float(parts[0]))
			cx, cy, w, h = map(float, parts[1:5])
			boxes.append((class_id, cx, cy, w, h))
		except ValueError:
			continue
	return boxes


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
	"""
	Convert normalized YOLO (cx, cy, w, h) to pixel (x1, y1, x2, y2).
	Clamps to image bounds.
	"""
	x1 = int(round((cx - w / 2.0) * img_w))
	y1 = int(round((cy - h / 2.0) * img_h))
	x2 = int(round((cx + w / 2.0) * img_w))
	y2 = int(round((cy + h / 2.0) * img_h))
	x1 = max(0, min(img_w - 1, x1))
	y1 = max(0, min(img_h - 1, y1))
	x2 = max(0, min(img_w - 1, x2))
	y2 = max(0, min(img_h - 1, y2))
	if x2 <= x1:
		x2 = min(img_w - 1, x1 + 1)
	if y2 <= y1:
		y2 = min(img_h - 1, y1 + 1)
	return x1, y1, x2, y2


def crop_with_fixed_ratio(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
	"""
	Crop a fixed jersey-focused region inside a bbox.
	"""
	x1, y1, x2, y2 = bbox
	h, w = img.shape[:2]
	bw = max(1, x2 - x1)
	bh = max(1, y2 - y1)

	x1r = int(round(x1 + 0.10 * bw))
	x2r = int(round(x2 - 0.10 * bw))
	y1r = int(round(y1 + 0.15 * bh))
	y2r = int(round(y1 + 0.65 * bh))

	x1r = max(0, min(w - 1, x1r))
	y1r = max(0, min(h - 1, y1r))
	x2r = max(0, min(w - 1, x2r))
	y2r = max(0, min(h - 1, y2r))
	if x2r <= x1r:
		x2r = min(w - 1, x1r + 1)
	if y2r <= y1r:
		y2r = min(h - 1, y1r + 1)
	return img[y1r:y2r, x1r:x2r]


def bgr_to_lab(roi_bgr: np.ndarray) -> np.ndarray:
	"""
	Convert BGR uint8 to OpenCV LAB uint8.
	"""
	return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)


def mask_non_field_pixels(
    roi_bgr: np.ndarray,
    roi_lab: np.ndarray,
    a_thresh: int = 118,
    chroma_thresh: int = 12,
    sat_thresh: int = 35,
    keep_upper_ratio: float = 0.60,
) -> np.ndarray:
	"""
	Strong green-screen style mask to remove likely field grass pixels.
	Combines LAB and HSV rules:
	- LAB: grass tends to have low 'a' (green-ish) and sufficient chroma.
	- HSV: grass tends to be green hue with moderate saturation.
	Returns mask where True means keep (non-field) pixels.
	"""
	a = roi_lab[:, :, 1].astype(np.int16)
	b = roi_lab[:, :, 2].astype(np.int16)
	chroma = np.abs(a - 128) + np.abs(b - 128)
	hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
	h = hsv[:, :, 0]
	s = hsv[:, :, 1]
	grass_lab = (a < a_thresh) & (chroma > chroma_thresh)
	grass_hsv = (h >= 25) & (h <= 95) & (s >= sat_thresh)
	grass_mask = grass_lab | grass_hsv
	keep_mask = ~grass_mask
	# Keep only the upper portion of non-field pixels to emphasize jersey area.
	h, _ = keep_mask.shape
	cutoff = int(round(max(0.0, min(1.0, keep_upper_ratio)) * h))
	if cutoff < h:
		keep_mask[cutoff:, :] = False
	# Fill tiny holes and remove isolated speckles for stability.
	k = np.ones((3, 3), np.uint8)
	keep_u8 = keep_mask.astype(np.uint8)
	keep_u8 = cv2.morphologyEx(keep_u8, cv2.MORPH_OPEN, k, iterations=1)
	keep_u8 = cv2.morphologyEx(keep_u8, cv2.MORPH_CLOSE, k, iterations=1)
	return keep_u8.astype(bool)


def lab_mean_std(roi_lab: np.ndarray, keep_mask: np.ndarray, min_pixels: int = 50) -> np.ndarray:
	"""
	Compute mean/std for L,a,b channels using a keep mask.
	Returns array of shape (6,): [meanL, meanA, meanB, stdL, stdA, stdB]
	"""
	if keep_mask is None or keep_mask.sum() < min_pixels:
		keep_mask = np.ones(roi_lab.shape[:2], dtype=bool)
	flat = roi_lab[keep_mask]
	if flat.size == 0:
		return np.zeros(6, dtype=np.float32)
	mean = flat.mean(axis=0)
	std = flat.std(axis=0)
	return np.concatenate([mean, std]).astype(np.float32)


def _effective_keep_mask(keep_mask: np.ndarray, min_pixels: int = 50) -> np.ndarray:
	"""
	Use full ROI if mask is too sparse so features remain stable.
	"""
	if keep_mask is None or int(keep_mask.sum()) < min_pixels:
		return np.ones_like(keep_mask, dtype=bool)
	return keep_mask


def _prepare_roi_and_mask(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Apply fixed jersey crop + non-field mask and return (roi_bgr, effective_keep_mask).
	"""
	h, w = img_bgr.shape[:2]
	roi = crop_with_fixed_ratio(img_bgr, (0, 0, w, h))
	lab = bgr_to_lab(roi)
	raw_mask = mask_non_field_pixels(roi, lab)
	keep_mask = _effective_keep_mask(raw_mask)
	return roi, keep_mask


def _render_kept_pixels(roi_bgr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
	"""
	Render only kept pixels, blacking out removed pixels.
	"""
	out = np.zeros_like(roi_bgr)
	out[keep_mask] = roi_bgr[keep_mask]
	return out


def save_processed_crop_previews(crop_paths: List[Path], out_dir: Path) -> int:
	"""
	Save per-player previews after jersey crop + green removal + upper keep ratio.
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	saved = 0
	for crop_path in crop_paths:
		img = cv2.imread(str(crop_path))
		if img is None:
			continue
		roi, keep_mask = _prepare_roi_and_mask(img)
		kept = _render_kept_pixels(roi, keep_mask)
		out_path = out_dir / crop_path.name
		if cv2.imwrite(str(out_path), kept):
			saved += 1
	print(f"[done] Saved {saved} processed crop previews to: {out_dir.resolve()}")
	return saved


def _iter_images(folder: Path) -> List[Path]:
	exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
	return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def compute_features(img_bgr: np.ndarray) -> np.ndarray:
	"""
	Run crop->LAB->mask->stats and return 6 values:
	[meanL, meanA, meanB, stdL, stdA, stdB]
	"""
	roi, keep_mask = _prepare_roi_and_mask(img_bgr)
	lab = bgr_to_lab(roi)
	return lab_mean_std(lab, keep_mask)


def cluster_and_copy(batch_dir: Path, out_parent: Path, k: int = 2) -> None:
	images = _iter_images(batch_dir)
	if not images:
		print(f"[warn] No images found in: {batch_dir}")
		return

	features = []
	valid_images = []
	for img_path in images:
		img = cv2.imread(str(img_path))
		if img is None:
			print(f"[warn] Cannot read image: {img_path}")
			continue
		feat = compute_features(img)
		features.append(feat.tolist())
		valid_images.append(img_path)

	if not valid_images:
		print(f"[warn] No readable images in: {batch_dir}")
		return

	out_parent.mkdir(parents=True, exist_ok=True)
	cluster_dirs = [out_parent / "cluster_0", out_parent / "cluster_1"]
	for d in cluster_dirs:
		d.mkdir(parents=True, exist_ok=True)

	if len(valid_images) == 1:
		dst = cluster_dirs[0] / valid_images[0].name
		shutil.copy2(valid_images[0], dst)
		print(f"{valid_images[0].name}\tcluster_0")
		return

	data = np.array(features, dtype=np.float32)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
	_, labels, _ = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
	labels = labels.flatten().tolist()

	for img_path, label in zip(valid_images, labels):
		label = int(label)
		dst = cluster_dirs[label] / img_path.name
		shutil.copy2(img_path, dst)
		print(f"{img_path.name}\tcluster_{label}")


def crop_boxes(image_path: Path, boxes_path: Path, out_dir: Path) -> List[Path]:
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Cannot read image: {image_path}")
	img_h, img_w = image.shape[:2]

	out_dir.mkdir(parents=True, exist_ok=True)
	boxes = _read_boxes(boxes_path)

	crop_paths: List[Path] = []
	for idx, (class_id, cx, cy, w, h) in enumerate(boxes):
		x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
		crop = image[y1:y2, x1:x2]
		if crop.size == 0:
			continue
		out_name = f"box_{idx:04d}_class{class_id}.jpg"
		out_path = out_dir / out_name
		cv2.imwrite(str(out_path), crop)
		crop_paths.append(out_path)
	return crop_paths


def cluster_crops(crop_paths: List[Path], out_parent: Path, k: int = 2) -> Tuple[Dict[int, int], Dict[int, List[float]]]:
	if not crop_paths:
		print("[warn] No crops to cluster.")
		return {}, {}

	features = []
	valid_images = []
	for img_path in crop_paths:
		img = cv2.imread(str(img_path))
		if img is None:
			print(f"[warn] Cannot read image: {img_path}")
			continue
		feat = compute_features(img)
		features.append(feat.tolist())
		valid_images.append(img_path)

	if not valid_images:
		print("[warn] No readable crops for clustering.")
		return {}, {}

	out_parent.mkdir(parents=True, exist_ok=True)
	cluster_dirs = [out_parent / "cluster_0", out_parent / "cluster_1"]
	for d in cluster_dirs:
		d.mkdir(parents=True, exist_ok=True)

	if len(valid_images) == 1:
		dst = cluster_dirs[0] / valid_images[0].name
		shutil.copy2(valid_images[0], dst)
		print(f"{valid_images[0].name}\tcluster_0")
		idx = _extract_box_index(valid_images[0])
		assignments = {idx: 0} if idx is not None else {}
		means = {0: list(features[0])} if features else {}
		return assignments, means

	data = np.array(features, dtype=np.float32)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
	_, labels, _ = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
	labels = labels.flatten().tolist()

	assignments: Dict[int, int] = {}
	feature_by_cluster: Dict[int, List[List[float]]] = {}
	for img_path, label in zip(valid_images, labels):
		label = int(label)
		dst = cluster_dirs[label] / img_path.name
		shutil.copy2(img_path, dst)
		print(f"{img_path.name}\tcluster_{label}")
		idx = _extract_box_index(img_path)
		if idx is not None:
			assignments[idx] = label
	for feat, label in zip(features, labels):
		label = int(label)
		feature_by_cluster.setdefault(label, []).append(feat)
	cluster_feature_means: Dict[int, List[float]] = {}
	for label, group in feature_by_cluster.items():
		arr = np.array(group, dtype=np.float32)
		cluster_feature_means[label] = arr.mean(axis=0).astype(np.float32).tolist()
	return assignments, cluster_feature_means


def main() -> None:
	ap = argparse.ArgumentParser(description="Crop boxes, compute LAB features, and cluster into 2 teams.")
	ap.add_argument("--image", required=True, help="Path to source image.")
	ap.add_argument("--boxes", required=True, help="Path to YOLO boxes TXT.")
	ap.add_argument("--out-dir", required=True, help="Output directory for crops and clusters.")
	ap.add_argument(
	    "--save-processed-crops",
	    action="store_true",
	    help="If set, save per-player previews after jersey crop + green removal to <out-dir>/processed_crops.",
	)
	args = ap.parse_args()

	out_dir = Path(args.out_dir)
	crops_dir = out_dir / "crops"
	crop_paths = crop_boxes(Path(args.image), Path(args.boxes), crops_dir)
	print(f"[done] Saved {len(crop_paths)} crops to: {crops_dir.resolve()}")
	if args.save_processed_crops:
		save_processed_crop_previews(crop_paths, out_dir / "processed_crops")

	assignments, cluster_feature_means = cluster_crops(crop_paths, out_dir)
	if assignments:
		_write_team_assignments_json(assignments, cluster_feature_means, out_dir)
	else:
		print("[warn] Team assignment JSON not written (no valid assignments).")


if __name__ == "__main__":
	main()
