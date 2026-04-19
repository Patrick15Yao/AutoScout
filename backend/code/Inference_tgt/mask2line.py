import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert YOLO-seg polygons in yardline.txt and harshmark.txt within a directory\n"
            "into fitted straight line segments (two endpoints per polygon) in normalized space [0,1].\n"
            "Outputs two files: yardline_line.txt and harshmark_line.txt."
        )
    )
    p.add_argument(
        "--txt-dir",
        required=True,
        help="Directory containing yardline.txt and/or harshmark.txt (YOLO-seg format).",
    )
    p.add_argument(
        "--out-dir",
        help="Optional output directory (default: same as --txt-dir).",
    )
    p.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Unused (reserved for potential future visualization).",
    )
    return p.parse_args()


def read_yolo_seg_txt(txt_path: Path) -> List[np.ndarray]:
    """
    Read a YOLO-segmentation polygon TXT and return a list of polygons.
    Each polygon is returned as an Nx2 array of normalized (x, y) coordinates.
    Lines not matching the expected format are skipped.
    """
    polygons: List[np.ndarray] = []
    if not txt_path.is_file():
        return polygons
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # Expect: class_id followed by pairs of coords (x y ...)
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                continue
            try:
                coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
            except Exception:
                continue
            polygons.append(coords)
    return polygons


def fit_line_segment(points_xy: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Fit a straight line to points (Nx2) using cv2.fitLine and
    return segment endpoints spanning the projections of the points.
    Returns None if fewer than two points are provided.
    """
    if points_xy.shape[0] < 2:
        return None
    pts = points_xy.astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    d = np.array([vx, vy], dtype=np.float32).reshape(2)
    p0 = np.array([x0, y0], dtype=np.float32).reshape(2)
    t = (pts - p0) @ d / (d @ d)
    tmin, tmax = t.min(), t.max()
    a = (p0 + tmin * d).astype(np.float32).tolist()
    b = (p0 + tmax * d).astype(np.float32).tolist()
    return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))


def polys_to_lines_txt(
    polys_norm: List[np.ndarray],
) -> List[Tuple[float, float, float, float]]:
    """
    Convert normalized polygons to line segment endpoints in normalized coords.
    Returns a list of (x1, y1, x2, y2) for each polygon that could be fit.
    """
    lines: List[Tuple[float, float, float, float]] = []
    for poly in polys_norm:
        seg = fit_line_segment(poly.astype(np.float32))
        if seg is None:
            continue
        (x1, y1), (x2, y2) = seg
        # clamp to [0,1] to preserve normalization
        x1 = float(np.clip(x1, 0.0, 1.0))
        y1 = float(np.clip(y1, 0.0, 1.0))
        x2 = float(np.clip(x2, 0.0, 1.0))
        y2 = float(np.clip(y2, 0.0, 1.0))
        lines.append((x1, y1, x2, y2))
    return lines


def write_lines_txt(out_path: Path, lines: List[Tuple[float, float, float, float]]) -> None:
    """
    Write line endpoints to a TXT file, one line per polygon:
    x1 y1 x2 y2
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in lines:
            f.write(f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")


def process_pair(txt_dir: Path, out_dir: Path) -> None:
    """
    Process yardline.txt and harshmark.txt from txt_dir and write
    yardline_line.txt and harshmark_line.txt to out_dir.
    Missing input files are skipped with a warning.
    """
    for name in ("yardline", "harshmark"):
        in_txt = txt_dir / f"{name}.txt"
        if not in_txt.exists():
            print(f"WARNING: missing {in_txt}, skipping.")
            continue
        polys = read_yolo_seg_txt(in_txt)
        lines = polys_to_lines_txt(polys)
        out_txt = out_dir / f"{name}_line.txt"
        write_lines_txt(out_txt, lines)
        print(f"{name}: polygons={len(polys)} lines_written={len(lines)} -> {out_txt}")


def main() -> None:
    args = parse_args()
    txt_dir = Path(args.txt_dir)
    out_dir = Path(args.out_dir) if args.out_dir else txt_dir

    if not txt_dir.is_dir():
        raise SystemExit(f"--txt-dir is not a directory: {txt_dir}")
    process_pair(txt_dir, out_dir)


if __name__ == "__main__":
    main()


