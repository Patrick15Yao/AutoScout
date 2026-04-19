"""
Lightweight loader for homography-related inputs.

Given an image path and a TXT directory (players, pocket, harshmark_line, yardline_line),
this module:
- captures image dimensions
- parses player centers
- parses pocket detections
- parses hashmark and yardline line segments, storing both endpoints and ax + by = c form
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


Point = Tuple[float, float]
YoloDet = Tuple[int, float, float, float, float]  # class, cx, cy, w, h (all normalized)

# Field geometry defaults (in yards)
FIELD_WIDTH_YARDS = 53.33
YARD_STEP = 5.0
# Default NFL hash distances from top sideline (yards). Adjust if needed.
HASH_Y_OFFSETS = [23.58, 29.75]
TEAM_COLORS = {
    0: (0, 102, 255),   # blue
    1: (255, 0, 0),     # red
}


@dataclass
class LineSegment:
	p1: Point
	p2: Point
	coeffs: Tuple[float, float, float]  # (a, b, c) for ax + by = c


@dataclass
class ParsedInputs:
	image_path: Path
	image_size: Tuple[int, int]  # (width, height)
	players: List[Point]
	pocket: List[YoloDet]
	harshmark_lines: List[LineSegment]
	yardline_lines: List[LineSegment]


def _read_image_size(image_path: Path) -> Tuple[int, int]:
	with Image.open(image_path) as im:
		return im.size  # (width, height)


def _parse_yolo_det_file(path: Path) -> List[YoloDet]:
	lines: List[YoloDet] = []
	if not path.exists():
		return lines
	for raw in path.read_text().splitlines():
		if not raw.strip():
			continue
		parts = raw.strip().split()
		if len(parts) != 5:
			continue
		cl, cx, cy, w, h = parts
		lines.append((int(float(cl)), float(cx), float(cy), float(w), float(h)))
	return lines


def _parse_player_centers(path: Path) -> List[Point]:
	dets = _parse_yolo_det_file(path)
	return [(cx, cy) for (_, cx, cy, _, _) in dets]


def _line_coeffs(p1: Point, p2: Point) -> Tuple[float, float, float]:
	"""
	Given two points, return (a, b, c) such that a*x + b*y = c defines the line.
	"""
	x1, y1 = p1
	x2, y2 = p2
	a = y1 - y2
	b = x2 - x1
	c = a * x1 + b * y1
	return (a, b, c)


def _parse_line_segments(path: Path) -> List[LineSegment]:
	segs: List[LineSegment] = []
	if not path.exists():
		return segs
	for raw in path.read_text().splitlines():
		if not raw.strip():
			continue
		parts = raw.strip().split()
		if len(parts) < 4:
			continue
		x1, y1, x2, y2 = map(float, parts[:4])
		p1, p2 = (x1, y1), (x2, y2)
		segs.append(LineSegment(p1=p1, p2=p2, coeffs=_line_coeffs(p1, p2)))
	return segs


def _intersect_lines(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> Point | None:
	"""
	Intersect two lines in ax + by = c form. Returns (x, y) or None if parallel/degenerate.
	"""
	a1, b1, c1c = c1
	a2, b2, c2c = c2
	den = a1 * b2 - a2 * b1
	if abs(den) < 1e-12:
		return None
	x = (c1c * b2 - c2c * b1) / den
	y = (a1 * c2c - a2 * c1c) / den
	return (x, y)


def compute_hash_yard_intersections(hashmark_lines: List[LineSegment], yardline_lines: List[LineSegment]) -> List[List[Point | None]]:
	"""
	Build a 2D grid of intersections: grid[hash_idx][yard_idx] = (x, y) or None.
	Prints how many valid intersection points were found.
	"""
	grid: List[List[Point | None]] = []
	count = 0
	for h in hashmark_lines:
		row: List[Point | None] = []
		for y in yardline_lines:
			pt = _intersect_lines(h.coeffs, y.coeffs)
			if pt is not None:
				count += 1
			row.append(pt)
		grid.append(row)
	print(f"[hom] intersections: {count} points from {len(hashmark_lines)} hashmarks x {len(yardline_lines)} yardlines")
	for hi, row in enumerate(grid):
		for yi, pt in enumerate(row):
			print(f"[hom] hash {hi} x yard {yi}: {pt}")
	return grid


def visualize_intersections(
    image_path: str | Path,
    grid: List[List[Point | None]],
    out_path: str | Path | None = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 6,
) -> Path:
	"""
	Draw intersection points as red dots on the image and save next to the source by default.
	Returns the output path.
	"""
	img_path = Path(image_path).expanduser().resolve()
	if out_path is None:
		out_path = img_path.with_name(f"{img_path.stem}_intersections{img_path.suffix}")
	else:
		out_path = Path(out_path).expanduser().resolve()

	with Image.open(img_path) as im:
		im = im.convert("RGB")
		draw = ImageDraw.Draw(im)
		w, h = im.size
		count = 0
		for row in grid:
			for pt in row:
				if pt is None:
					continue
				xn, yn = pt
				xp = xn * w
				yp = yn * h
				draw.ellipse(
				    (xp - radius, yp - radius, xp + radius, yp + radius),
				    fill=color,
				    outline=color,
				)
				count += 1
		im.save(out_path)
	print(f"[hom] saved {count} intersections to {out_path}")
	return out_path


def _safe_intersection_x(ref_coeffs: Tuple[float, float, float], seg: LineSegment) -> float:
	pt = _intersect_lines(ref_coeffs, seg.coeffs)
	return pt[0] if pt is not None else float("inf")


def order_lines_by_intersections(
    hashmark_lines: List[LineSegment],
    yardline_lines: List[LineSegment],
) -> Tuple[List[LineSegment], List[LineSegment]]:
	"""
	Reorder hashmarks by vertical (y) position, then reorder yardlines by intersection
	x with the first (top) hashmark. Returns (sorted_hashmarks, sorted_yardlines).
	"""
	if not hashmark_lines or not yardline_lines:
		return hashmark_lines, yardline_lines

	def mid_y(seg: LineSegment) -> float:
		return (seg.p1[1] + seg.p2[1]) / 2.0

	sorted_hashes = sorted(hashmark_lines, key=mid_y)
	ref_hash = sorted_hashes[0]
	sorted_yards = sorted(yardline_lines, key=lambda seg: _safe_intersection_x(ref_hash.coeffs, seg))

	ref_yard = sorted_yards[0] if sorted_yards else None
	if ref_yard is None:
		return sorted_hashes, sorted_yards

	return sorted_hashes, sorted_yards


def build_field_grid(
    num_hash: int,
    num_yard: int,
    yard_step: float = YARD_STEP,
    field_width: float = FIELD_WIDTH_YARDS,
    hash_offsets: List[float] | None = None,
) -> List[List[Tuple[float, float]]]:
	"""
	Construct a field coordinate grid aligning with the intersection grid:
	field[hash_idx][yard_idx] = (X_field, Y_field).
	- Yardlines are spaced by yard_step (default 5 yards).
	- Hash Y offsets default to HASH_Y_OFFSETS if lengths match; otherwise spread evenly between 0 and field_width.
	"""
	if hash_offsets is None:
		hash_offsets = HASH_Y_OFFSETS
	if hash_offsets and len(hash_offsets) == num_hash:
		y_vals = list(hash_offsets)
	elif num_hash > 1:
		# Evenly distribute between top and bottom sideline
		y_vals = [field_width * i / (num_hash - 1) for i in range(num_hash)]
	else:
		y_vals = [field_width / 2.0] * max(1, num_hash)

	grid: List[List[Tuple[float, float]]] = []
	for hi in range(num_hash):
		row: List[Tuple[float, float]] = []
		for yi in range(num_yard):
			x = yi * yard_step
			y = y_vals[hi]
			row.append((x, y))
		grid.append(row)
	return grid


def compute_homography_from_grid(
    img_grid: List[List[Point | None]],
    field_grid: List[List[Tuple[float, float]]],
    ransac_thresh: float = 3.0,
) -> Tuple[np.ndarray | None, np.ndarray | None, int]:
	"""
	Compute homography using corresponding points from image grid and field grid.
	Returns (H, mask, inlier_count).
	"""
	img_pts = []
	field_pts = []
	for hi, row in enumerate(img_grid):
		for yi, pt in enumerate(row):
			if pt is None:
				continue
			if hi >= len(field_grid) or yi >= len(field_grid[hi]):
				continue
			img_pts.append(pt)
			field_pts.append(field_grid[hi][yi])
	if len(img_pts) < 4:
		print(f"[hom] not enough points for homography: {len(img_pts)} (<4)")
		return None, None, 0

	img_arr = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
	field_arr = np.array(field_pts, dtype=np.float32).reshape(-1, 1, 2)

	H, mask = cv2.findHomography(img_arr, field_arr, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
	inliers = int(mask.sum()) if mask is not None else 0
	if H is None:
		print("[hom] homography failed (H is None)")
	else:
		print(f"[hom] homography computed with {inliers}/{len(img_pts)} inliers")
	return H, mask, inliers


def apply_homography_to_points(points: List[Point], H: np.ndarray | None) -> List[Point | None]:
	"""
	Apply homography H to a list of normalized image points.
	Returns list of mapped points (None if H is None).
	"""
	if H is None or len(points) == 0:
		return [None for _ in points]
	arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
	mapped = cv2.perspectiveTransform(arr, H).reshape(-1, 2)
	return [(float(x), float(y)) for x, y in mapped]


def load_team_assignments(team_json_path: str | Path, num_players: int) -> List[int | None]:
	"""
	Load team assignments from JSON and return a list aligned with players.txt row indices.
	Each entry is cluster id (0/1) or None when missing.
	"""
	path = Path(team_json_path).expanduser().resolve()
	if not path.exists():
		print(f"[hom] team JSON not found: {path}")
		return [None] * num_players
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		print(f"[hom] failed to parse team JSON {path}: {exc}")
		return [None] * num_players

	labels: List[int | None] = [None] * num_players
	items = payload.get("assignments", [])
	if not isinstance(items, list):
		print(f"[hom] invalid assignments format in: {path}")
		return labels

	assigned = 0
	for item in items:
		if not isinstance(item, dict):
			continue
		idx = item.get("player_index")
		team = item.get("team_cluster")
		if not isinstance(idx, int) or not isinstance(team, int):
			continue
		if 0 <= idx < num_players:
			labels[idx] = team
			assigned += 1
	print(f"[hom] loaded team labels: {assigned}/{num_players} from {path}")
	return labels


def _draw_dotted_line(draw: ImageDraw.ImageDraw, p1: Tuple[float, float], p2: Tuple[float, float], dash: int = 8, gap: int = 6, **kwargs) -> None:
	x1, y1 = p1
	x2, y2 = p2
	dx = x2 - x1
	dy = y2 - y1
	dist = (dx * dx + dy * dy) ** 0.5
	if dist == 0:
		return
	steps = int(dist // (dash + gap)) + 1
	for i in range(steps):
		start_ratio = i * (dash + gap) / dist
		end_ratio = min(start_ratio + dash / dist, 1.0)
		sx = x1 + dx * start_ratio
		sy = y1 + dy * start_ratio
		ex = x1 + dx * end_ratio
		ey = y1 + dy * end_ratio
		draw.line((sx, sy, ex, ey), **kwargs)


def draw_field_topdown(
    players_field: List[Point | None],
    team_labels: List[int | None] | None,
    player_texts: List[str | None] | None,
    only_labeled: bool,
    hash_y: List[float],
    yard_x: List[float],
    out_path: str | Path,
    field_width: float = FIELD_WIDTH_YARDS,
    pad: int = 20,
    canvas_w: int = 1200,
    canvas_h: int = 600,
) -> Path:
	"""
	Draw a top-down field with yardlines/hash lines and player positions.
	- Adds one extra yardline before the first and after the last (sidelines).
	- Uses y=0 and y=field_width for top/bottom sidelines.
	- Players drawn by team color when team labels are provided.
	- Optional player_texts are rendered in each dot (e.g., jersey number 1..11).
	- Hash lines dotted white, yardlines solid white.
	"""
	if not yard_x:
		return Path(out_path)
	xs = list(yard_x)
	step = xs[1] - xs[0] if len(xs) > 1 else YARD_STEP
	x_min = xs[0] - step
	x_max = xs[-1] + step

	y_top = 0.0
	y_bottom = field_width

	yard_positions = [x_min] + xs + [x_max]

	scale_x = (canvas_w - 2 * pad) / (x_max - x_min)
	scale_y = (canvas_h - 2 * pad) / (y_bottom - y_top)

	def to_px(pt: Tuple[float, float]) -> Tuple[float, float]:
		x, y = pt
		px = pad + (x - x_min) * scale_x
		py = pad + (y - y_top) * scale_y
		return (px, py)

	img = Image.new("RGB", (canvas_w, canvas_h), color=(20, 90, 20))
	draw = ImageDraw.Draw(img)

	for x in yard_positions:
		p1 = to_px((x, y_top))
		p2 = to_px((x, y_bottom))
		draw.line((*p1, *p2), fill=(255, 255, 255), width=2)

	for y in hash_y:
		p1 = to_px((x_min, y))
		p2 = to_px((x_max, y))
		_draw_dotted_line(draw, p1, p2, fill=(255, 255, 255), width=2)

	for idx, pt in enumerate(players_field):
		if pt is None:
			continue
		label = player_texts[idx] if player_texts is not None and idx < len(player_texts) else None
		if only_labeled and not label:
			continue
		px, py = to_px(pt)
		r = 8
		team = team_labels[idx] if team_labels is not None and idx < len(team_labels) else None
		color = TEAM_COLORS.get(team, (255, 0, 0))
		draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=color)
		if label:
			draw.text((px, py), str(label), fill=(255, 255, 255), anchor="mm")

	out_path = Path(out_path).expanduser().resolve()
	img.save(out_path)
	print(f"[hom] saved field view to: {out_path}")
	return out_path


def load_inputs(image_path: str | Path, txt_dir: str | Path) -> ParsedInputs:
	image_path = Path(image_path).expanduser().resolve()
	txt_dir = Path(txt_dir).expanduser().resolve()

	players_txt = txt_dir / "players.txt"
	pocket_txt = txt_dir / "pocket.txt"
	harshmark_txt = txt_dir / "harshmark_line.txt"
	yardline_txt = txt_dir / "yardline_line.txt"

	image_size = _read_image_size(image_path)
	players = _parse_player_centers(players_txt)
	pocket = _parse_yolo_det_file(pocket_txt)
	harshmark_lines = _parse_line_segments(harshmark_txt)
	yardline_lines = _parse_line_segments(yardline_txt)

	return ParsedInputs(
	    image_path=image_path,
	    image_size=image_size,
	    players=players,
	    pocket=pocket,
	    harshmark_lines=harshmark_lines,
	    yardline_lines=yardline_lines,
	)


def load_player_texts(players_label_txt: str | Path, num_players: int) -> List[str | None]:
	"""
	Load optional label texts aligned by players.txt row index.
	Expected line format:
	player_index team_id team_slot track_id
	Only team_slot is rendered.
	"""
	path = Path(players_label_txt).expanduser().resolve()
	out: List[str | None] = [None] * num_players
	if not path.exists():
		return out
	for raw in path.read_text(encoding="utf-8").splitlines():
		parts = raw.strip().split()
		if len(parts) < 3:
			continue
		try:
			idx = int(parts[0])
			team_slot = int(parts[2])
		except Exception:
			continue
		if 0 <= idx < num_players and team_slot > 0:
			out[idx] = str(team_slot)
	return out


__all__ = [
    "ParsedInputs",
    "LineSegment",
    "load_inputs",
    "compute_hash_yard_intersections",
    "visualize_intersections",
    "order_lines_by_intersections",
    "build_field_grid",
    "compute_homography_from_grid",
    "apply_homography_to_points",
    "draw_field_topdown",
]


def main() -> None:
	import argparse

	ap = argparse.ArgumentParser(description="Load homography inputs and report intersections.")
	ap.add_argument("--image", required=True, help="Path to source image.")
	ap.add_argument("--txt-dir", required=True, help="Directory containing players.txt, pocket.txt, harshmark_line.txt, yardline_line.txt.")
	ap.add_argument("--out", help="Optional output path for visualization. Defaults to <image_stem>_intersections.<ext> next to the image.")
	ap.add_argument("--team-json", help="Optional team assignment JSON (from Color_Clustering). Defaults to ../Indi/team_assignments.json from --txt-dir.")
	ap.add_argument("--players-label-txt", help="Optional players label txt for rendering numbers in field dots.")
	ap.add_argument("--only-labeled", action="store_true", help="If set, draw only players that have a non-empty label.")
	args = ap.parse_args()

	data = load_inputs(args.image, args.txt_dir)
	print(f"[hom] image: {data.image_path.name}, size: {data.image_size}")
	print(f"[hom] players: {len(data.players)}, pocket dets: {len(data.pocket)}")
	print(f"[hom] hash lines: {len(data.harshmark_lines)}, yard lines: {len(data.yardline_lines)}")

	hash_sorted, yard_sorted = order_lines_by_intersections(data.harshmark_lines, data.yardline_lines)
	print(f"[hom] sorted yardlines by hash0 intersection x")
	print(f"[hom] sorted hashmarks by yard0 intersection x")

	grid = compute_hash_yard_intersections(hash_sorted, yard_sorted)
	field_grid = build_field_grid(len(hash_sorted), len(yard_sorted))
	if grid and grid[0]:
		print(f"[hom] sample intersection [0][0]: {grid[0][0]}")
		print("[hom] image coords -> field coords:")
		for hi, row in enumerate(grid):
			for yi, pt in enumerate(row):
				fpt = field_grid[hi][yi] if hi < len(field_grid) and yi < len(field_grid[hi]) else None
				print(f"  hash {hi} x yard {yi}: img={pt} -> field={fpt}")
	H, mask, inliers = compute_homography_from_grid(grid, field_grid)
	if H is not None:
		print("[hom] homography matrix:")
		print(H)
		print(f"[hom] inliers: {inliers}")
		player_field = apply_homography_to_points(data.players, H)
		print("[hom] player coords (img -> field):")
		for i, (p_img, p_field) in enumerate(zip(data.players, player_field)):
			print(f"  player {i}: img={p_img} -> field={p_field}")
		txt_dir = Path(args.txt_dir).expanduser().resolve()
		default_team_json = txt_dir.parent / "Indi" / "team_assignments.json"
		team_json = Path(args.team_json).expanduser().resolve() if args.team_json else default_team_json
		team_labels = load_team_assignments(team_json, len(player_field))
		player_texts = (
		    load_player_texts(args.players_label_txt, len(player_field))
		    if args.players_label_txt
		    else [None] * len(player_field)
		)
		# Build yard/hash axes for field rendering
		yard_x = [pt[0] for pt in field_grid[0]] if field_grid and field_grid[0] else []
		hash_y = [row[0][1] for row in field_grid if row]
		base_dir = Path(args.out).expanduser().resolve().parent if args.out else data.image_path.parent
		field_out = base_dir / f"{data.image_path.stem}_field{data.image_path.suffix}"
		draw_field_topdown(player_field, team_labels, player_texts, bool(args.only_labeled), hash_y, yard_x, field_out)
	out_path = visualize_intersections(args.image, grid, out_path=args.out)
	print(f"[hom] visualization saved to: {out_path}")


if __name__ == "__main__":
	main()

