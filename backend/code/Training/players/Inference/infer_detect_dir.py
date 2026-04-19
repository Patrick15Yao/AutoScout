#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Set

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run YOLO detection on all JPGs in a directory and save white-box JPGs + YOLO txt labels."
	)
	parser.add_argument("--weights", type=str, required=True, help="Path to YOLO detection weights (.pt).")
	parser.add_argument("--source", type=str, required=True, help="Input images directory (jpg/jpeg).")
	parser.add_argument("--out-dir", type=str, required=True, help="Output directory for JPG + TXT.")
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
	parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or CUDA index like '0'.")
	parser.add_argument("--line-width", type=int, default=2, help="Box line width in pixels.")
	return parser.parse_args()


def list_images(directory: Path) -> List[Path]:
	return sorted([p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()])


def draw_white_boxes(orig_bgr, xyxy, line_width: int) -> None:
	# xyxy: Nx4 tensor/array in pixels
	if xyxy is None:
		return
	for box in xyxy:
		x1, y1, x2, y2 = map(int, box)
		cv2.rectangle(orig_bgr, (x1, y1), (x2, y2), (255, 255, 255), thickness=line_width)


def main() -> None:
	args = parse_args()

	weights_path = Path(args.weights).expanduser().resolve()
	if not weights_path.exists():
		raise SystemExit(f"[error] Weights not found: {weights_path}")

	source_dir = Path(args.source).expanduser().resolve()
	if not source_dir.exists():
		raise SystemExit(f"[error] Source directory not found: {source_dir}")

	out_dir = Path(args.out_dir).expanduser().resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	images = list_images(source_dir)
	if not images:
		raise SystemExit(f"[error] No JPG images found under: {source_dir}")

	model = YOLO(str(weights_path))
	results = model.predict(
		source=[str(p) for p in images],
		device=args.device,
		conf=args.conf,
		verbose=True,
	)

	saved = 0
	for r in results:
		# Build visualization with white boxes, no labels
		img = r.orig_img.copy()  # BGR
		xyxy = getattr(r.boxes, "xyxy", None).cpu().numpy() if getattr(r, "boxes", None) is not None else None
		draw_white_boxes(img, xyxy, args.line_width)
		out_jpg = out_dir / Path(r.path).name
		if cv2.imwrite(str(out_jpg), img):
			saved += 1

		# Write YOLO detection labels: <class> <cx> <cy> <w> <h> (normalized)
		out_txt = out_dir / (Path(r.path).stem + ".txt")
		lines: List[str] = []
		if getattr(r, "boxes", None) is not None and getattr(r.boxes, "xywhn", None) is not None:
			xywhn = r.boxes.xywhn.tolist()
			classes = r.boxes.cls.tolist() if getattr(r.boxes, "cls", None) is not None else [0] * len(xywhn)
			for (cx, cy, w, h), c in zip(xywhn, classes):
				lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
		out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))

	print(f"[done] Saved {saved}/{len(results)} JPGs and TXT labels to: {out_dir}")


if __name__ == "__main__":
	main()

