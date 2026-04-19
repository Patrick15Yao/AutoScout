#!/usr/bin/env python3
"""
Video-level pipeline runner:
- sample frames from input video at target FPS
- run per-frame inference to generate txt outputs
- optionally run jersey color clustering (Indi) and visualization outputs
- always run homography field projection per frame

Output layout:
<out_dir>/
  frames/
    frame_000000/
      frame_000000.jpg
      txt/                     # official output
      frame_000000_field.jpg   # official output
  logs/
    indi/...                   # optional
    vis/...                    # optional
    intersections/...          # optional
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

try:
	import supervision as sv
except Exception:
	sv = None

try:
	from Homography.hom import (
	    apply_homography_to_points,
	    build_field_grid,
	    compute_hash_yard_intersections,
	    compute_homography_from_grid,
	    load_inputs,
	    order_lines_by_intersections,
	)
except Exception:
	load_inputs = None
	order_lines_by_intersections = None
	compute_hash_yard_intersections = None
	build_field_grid = None
	compute_homography_from_grid = None
	apply_homography_to_points = None


ROOT = Path(__file__).resolve().parents[1]
INFER_SCRIPT = ROOT / "code" / "Inference_tgt" / "inference_all.py"
VIS_SCRIPT = ROOT / "code" / "Inference_tgt" / "Visualize_1.py"
CLUSTER_SCRIPT = ROOT / "code" / "ColorLab" / "Color_Clustering.py"
HOM_SCRIPT = ROOT / "code" / "Homography" / "hom.py"
DEFAULT_CONFIG = ROOT / "code" / "Inference_tgt" / "para_local.json"


def run(cmd: list[str]) -> None:
	print("[run]", " ".join(str(c) for c in cmd))
	subprocess.run(cmd, check=True)


def _load_team_json(path: Path) -> dict:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return {}


def _extract_cluster_means(payload: dict) -> dict[int, list[float]]:
	out: dict[int, list[float]] = {}
	items = payload.get("cluster_feature_means", [])
	if not isinstance(items, list):
		return out
	for item in items:
		if not isinstance(item, dict):
			continue
		c = item.get("team_cluster")
		feat = item.get("mean_feature")
		if isinstance(c, int) and isinstance(feat, list) and feat:
			try:
				out[c] = [float(x) for x in feat]
			except Exception:
				continue
	return out


def _l2(a: list[float], b: list[float]) -> float:
	if len(a) != len(b):
		return float("inf")
	return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _choose_cluster_mapping(cur_means: dict[int, list[float]], ref_means: dict[int, list[float]]) -> dict[int, int]:
	"""
	Map current frame cluster ids -> canonical cluster ids from first frame.
	Supports normal 2-cluster case and single-cluster fallback.
	"""
	if not cur_means or not ref_means:
		return {}
	cur_keys = sorted(cur_means.keys())
	ref_keys = sorted(ref_means.keys())
	if len(cur_keys) == 1 and len(ref_keys) >= 1:
		cur = cur_keys[0]
		best_ref = min(ref_keys, key=lambda r: _l2(cur_means[cur], ref_means[r]))
		return {cur: best_ref}
	if len(cur_keys) < 2 or len(ref_keys) < 2:
		return {}
	c0, c1 = cur_keys[0], cur_keys[1]
	r0, r1 = ref_keys[0], ref_keys[1]
	d_identity = _l2(cur_means[c0], ref_means[r0]) + _l2(cur_means[c1], ref_means[r1])
	d_swap = _l2(cur_means[c0], ref_means[r1]) + _l2(cur_means[c1], ref_means[r0])
	if d_identity <= d_swap:
		return {c0: r0, c1: r1}
	return {c0: r1, c1: r0}


def _rewrite_team_json_with_mapping(path: Path, mapping: dict[int, int]) -> None:
	if not mapping:
		return
	payload = _load_team_json(path)
	changed = 0
	items = payload.get("assignments", [])
	if isinstance(items, list):
		for item in items:
			if not isinstance(item, dict):
				continue
			c = item.get("team_cluster")
			if isinstance(c, int) and c in mapping:
				item["team_cluster"] = mapping[c]
				changed += 1
	means = payload.get("cluster_feature_means", [])
	if isinstance(means, list):
		for item in means:
			if not isinstance(item, dict):
				continue
			c = item.get("team_cluster")
			if isinstance(c, int) and c in mapping:
				item["team_cluster"] = mapping[c]
	payload["temporal_consistency_remap"] = {str(k): v for k, v in mapping.items()}
	path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
	if changed:
		print(f"[info] remapped {changed} player labels for temporal consistency: {path}")


def _load_team_assignments_map(path: Path) -> dict[int, int]:
	payload = _load_team_json(path)
	out: dict[int, int] = {}
	items = payload.get("assignments", [])
	if not isinstance(items, list):
		return out
	for item in items:
		if not isinstance(item, dict):
			continue
		idx = item.get("player_index")
		team = item.get("team_cluster")
		if isinstance(idx, int) and isinstance(team, int):
			out[idx] = team
	return out


def _write_players_team_txt(players_txt: Path, team_json_path: Path, out_path: Path) -> None:
	"""
	Write players_team.txt in format:
	class cx cy w h team_id
	where team_id is temporally-consistent when available, else -1.
	"""
	if not players_txt.exists() or not team_json_path.exists():
		return
	assignments = _load_team_assignments_map(team_json_path)
	lines = players_txt.read_text(encoding="utf-8").splitlines()
	out_lines: list[str] = []
	for i, line in enumerate(lines):
		parts = line.strip().split()
		if len(parts) < 5:
			continue
		team_id = assignments.get(i, -1)
		out_lines.append(" ".join(parts[:5] + [str(team_id)]))
	out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
	print(f"[done] wrote team txt: {out_path}")


def _count_teams_from_players_team_txt(players_team_txt: Path) -> tuple[int, int, int, int]:
	"""
	Count total/team0/team1/unknown from players_team.txt lines:
	class cx cy w h team_id
	"""
	total = 0
	team0 = 0
	team1 = 0
	unknown = 0
	if not players_team_txt.exists():
		return total, team0, team1, unknown
	for raw in players_team_txt.read_text(encoding="utf-8").splitlines():
		parts = raw.strip().split()
		if len(parts) < 6:
			continue
		total += 1
		try:
			team_id = int(float(parts[5]))
		except Exception:
			unknown += 1
			continue
		if team_id == 0:
			team0 += 1
		elif team_id == 1:
			team1 += 1
		else:
			unknown += 1
	return total, team0, team1, unknown


def _load_players_team_rows(players_team_txt: Path) -> list[dict]:
	rows: list[dict] = []
	if not players_team_txt.exists():
		return rows
	for i, raw in enumerate(players_team_txt.read_text(encoding="utf-8").splitlines()):
		parts = raw.strip().split()
		if len(parts) < 6:
			continue
		try:
			cl = int(float(parts[0]))
			cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
			team = int(float(parts[5]))
		except Exception:
			continue
		rows.append(
		    {"idx": i, "class_id": cl, "cx": cx, "cy": cy, "w": w, "h": h, "team": team}
		)
	return rows


def _load_players_conf_rows(players_conf_txt: Path) -> list[dict]:
	rows: list[dict] = []
	if not players_conf_txt.exists():
		return rows
	for i, raw in enumerate(players_conf_txt.read_text(encoding="utf-8").splitlines()):
		parts = raw.strip().split()
		if len(parts) < 6:
			continue
		try:
			cl = int(float(parts[0]))
			cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4]); conf = float(parts[5])
		except Exception:
			continue
		rows.append(
		    {"idx": i, "class_id": cl, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf}
		)
	return rows


def _write_players_team_rows(players_team_txt: Path, rows: list[dict]) -> None:
	lines: list[str] = []
	for r in rows:
		lines.append(
		    f"{int(r['class_id'])} {float(r['cx']):.6f} {float(r['cy']):.6f} "
		    f"{float(r['w']):.6f} {float(r['h']):.6f} {int(r['team'])}"
		)
	players_team_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _track_all_with_bytetrack(
    det_rows: list[dict],
    tracker,
    img_w: int,
    img_h: int,
) -> dict[int, int]:
	"""
	Global ByteTrack over all player detections in a frame.
	Returns mapping: player_index -> track_id.
	"""
	n = len(det_rows)
	if n == 0:
		return {}
	xyxy = []
	conf = []
	for d in det_rows:
		x1n, y1n, x2n, y2n = _xyxy_from_cxcywh(float(d["cx"]), float(d["cy"]), float(d["w"]), float(d["h"]))
		xyxy.append([x1n * img_w, y1n * img_h, x2n * img_w, y2n * img_h])
		conf.append(float(d.get("conf", 0.01)))
	det_idx = np.arange(n, dtype=np.int32)
	detections = sv.Detections(
	    xyxy=np.array(xyxy, dtype=np.float32),
	    confidence=np.array(conf, dtype=np.float32),
	    class_id=np.zeros(n, dtype=np.int32),
	    data={"det_idx": det_idx},
	)
	tracked = tracker.update_with_detections(detections)
	out: dict[int, int] = {}
	try:
		t_det_idx = tracked.data.get("det_idx")
		t_track_id = tracked.tracker_id
		if t_det_idx is not None and t_track_id is not None:
			for di, tid in zip(t_det_idx.tolist(), t_track_id.tolist()):
				di = int(di); tid = int(tid)
				if 0 <= di < n:
					out[int(det_rows[di]["idx"])] = tid
	except Exception:
		pass
	return out


def _stabilize_team_rows_by_track(
    players_team_rows: list[dict],
    conf_rows: list[dict],
    det_track_map: dict[int, int],
    track_team_score: dict[int, float],
) -> list[dict]:
	"""
	Track-level team stabilization:
	- update per-track running score from current frame team labels
	- assign stabilized team from running score when available
	"""
	team_by_idx = {int(r["idx"]): int(r["team"]) for r in players_team_rows}
	out_rows: list[dict] = []
	alpha = 0.9  # EMA memory
	for d in conf_rows:
		idx = int(d["idx"])
		raw_team = team_by_idx.get(idx, -1)
		tid = int(det_track_map.get(idx, -1))
		score = None
		if tid > 0:
			old = float(track_team_score.get(tid, 0.0))
			if raw_team in (0, 1):
				signal = 1.0 if raw_team == 1 else -1.0
				score = alpha * old + (1.0 - alpha) * signal * max(0.2, min(1.0, float(d.get("conf", 0.5))))
			else:
				score = old
			track_team_score[tid] = score
		if score is None and tid > 0:
			score = float(track_team_score.get(tid, 0.0))
		if score is not None and abs(score) >= 0.05:
			stable_team = 1 if score > 0 else 0
		elif raw_team in (0, 1):
			stable_team = raw_team
		else:
			stable_team = 0
		out_rows.append(
		    {
		        "idx": idx,
		        "class_id": int(d["class_id"]),
		        "cx": float(d["cx"]),
		        "cy": float(d["cy"]),
		        "w": float(d["w"]),
		        "h": float(d["h"]),
		        "conf": float(d.get("conf", 0.0)),
		        "team": int(stable_team),
		        "track_id": tid,
		    }
		)
	return out_rows


def _load_conf_by_index(players_conf_txt: Path) -> dict[int, float]:
	out: dict[int, float] = {}
	if not players_conf_txt.exists():
		return out
	for i, raw in enumerate(players_conf_txt.read_text(encoding="utf-8").splitlines()):
		parts = raw.strip().split()
		if len(parts) < 6:
			continue
		try:
			out[i] = float(parts[5])
		except Exception:
			continue
	return out


def _load_pocket_boxes(pocket_txt: Path) -> list[tuple[float, float, float, float]]:
	"""
	Load pocket boxes from YOLO det lines and convert to normalized xyxy.
	"""
	boxes: list[tuple[float, float, float, float]] = []
	if not pocket_txt.exists():
		return boxes
	for raw in pocket_txt.read_text(encoding="utf-8").splitlines():
		parts = raw.strip().split()
		if len(parts) < 5:
			continue
		try:
			cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
		except Exception:
			continue
		x1 = max(0.0, cx - w / 2.0)
		y1 = max(0.0, cy - h / 2.0)
		x2 = min(1.0, cx + w / 2.0)
		y2 = min(1.0, cy + h / 2.0)
		boxes.append((x1, y1, x2, y2))
	return boxes


def _is_inside_any_pocket(cx: float, cy: float, pockets: list[tuple[float, float, float, float]]) -> bool:
	for x1, y1, x2, y2 in pockets:
		if x1 <= cx <= x2 and y1 <= cy <= y2:
			return True
	return False


def _select_players_with_pocket_priority(
    players_team_txt: Path,
    players_conf_txt: Path,
    pocket_txt: Path,
    target_per_team: int = 11,
) -> dict:
	"""
	Select players for each team:
	1) prioritize players outside pocket
	2) fill missing slots from inside-pocket players by confidence
	"""
	rows = _load_players_team_rows(players_team_txt)
	conf_by_idx = _load_conf_by_index(players_conf_txt)
	pockets = _load_pocket_boxes(pocket_txt)

	for r in rows:
		r["conf"] = float(conf_by_idx.get(int(r["idx"]), 0.0))
		r["in_pocket"] = _is_inside_any_pocket(float(r["cx"]), float(r["cy"]), pockets)

	selected_rows: list[dict] = []
	stats = {
	    "raw_total": len(rows),
	    "raw_team0": sum(1 for r in rows if r["team"] == 0),
	    "raw_team1": sum(1 for r in rows if r["team"] == 1),
	    "raw_unknown": sum(1 for r in rows if r["team"] not in (0, 1)),
	}

	for team in (0, 1):
		team_rows = [r for r in rows if r["team"] == team]
		outside = sorted([r for r in team_rows if not r["in_pocket"]], key=lambda x: x["conf"], reverse=True)
		inside = sorted([r for r in team_rows if r["in_pocket"]], key=lambda x: x["conf"], reverse=True)
		keep_outside = outside[:target_per_team]
		need = max(0, target_per_team - len(keep_outside))
		keep_inside = inside[:need]
		selected_rows.extend(keep_outside + keep_inside)
		stats[f"selected_team{team}"] = len(keep_outside) + len(keep_inside)

	stats["selected_total"] = len(selected_rows)
	stats["selected_unknown"] = 0
	stats["target_per_team"] = target_per_team
	stats["selected_rows"] = selected_rows
	return stats


def _write_players_team_selected_txt(out_path: Path, selected_rows: list[dict]) -> None:
	"""
	Write selected players with team id, confidence, and pocket flag:
	class cx cy w h team_id conf in_pocket
	"""
	lines: list[str] = []
	for r in selected_rows:
		lines.append(
		    f"{int(r['class_id'])} {r['cx']:.6f} {r['cy']:.6f} {r['w']:.6f} {r['h']:.6f} "
		    f"{int(r['team'])} {float(r['conf']):.6f} {int(bool(r['in_pocket']))}"
		)
	out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
	print(f"[done] wrote pocket-priority team txt: {out_path}")


def _bbox_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1 = max(ax1, bx1)
	iy1 = max(ay1, by1)
	ix2 = min(ax2, bx2)
	iy2 = min(ay2, by2)
	iw = max(0.0, ix2 - ix1)
	ih = max(0.0, iy2 - iy1)
	inter = iw * ih
	aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = aa + ba - inter
	if union <= 0:
		return 0.0
	return inter / union


def _xyxy_from_cxcywh(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
	x1 = max(0.0, cx - w / 2.0)
	y1 = max(0.0, cy - h / 2.0)
	x2 = min(1.0, cx + w / 2.0)
	y2 = min(1.0, cy + h / 2.0)
	return (x1, y1, x2, y2)


def _track_team_with_bytetrack(
    dets: list[dict],
    tracker,
    img_w: int,
    img_h: int,
) -> list[int]:
	"""
	Run ByteTrack for one team's detections and return track IDs aligned with dets.
	"""
	n = len(dets)
	if n == 0:
		return []

	xyxy = []
	conf = []
	for d in dets:
		x1n, y1n, x2n, y2n = _xyxy_from_cxcywh(float(d["cx"]), float(d["cy"]), float(d["w"]), float(d["h"]))
		xyxy.append([x1n * img_w, y1n * img_h, x2n * img_w, y2n * img_h])
		conf.append(float(d.get("conf", 0.0)))

	det_idx = np.arange(n, dtype=np.int32)
	detections = sv.Detections(
	    xyxy=np.array(xyxy, dtype=np.float32),
	    confidence=np.array(conf, dtype=np.float32),
	    class_id=np.zeros(n, dtype=np.int32),
	    data={"det_idx": det_idx},
	)
	tracked = tracker.update_with_detections(detections)
	track_ids = [-1] * n

	# Preferred mapping: preserved det_idx metadata.
	try:
		t_det_idx = tracked.data.get("det_idx")
		t_track_id = tracked.tracker_id
		if t_det_idx is not None and t_track_id is not None:
			for di, tid in zip(t_det_idx.tolist(), t_track_id.tolist()):
				di = int(di)
				tid = int(tid)
				if 0 <= di < n:
					track_ids[di] = tid
	except Exception:
		pass

	# Fallback mapping by IoU between input dets and tracked boxes.
	if any(tid < 0 for tid in track_ids):
		used_tr = set()
		t_boxes = tracked.xyxy.tolist() if getattr(tracked, "xyxy", None) is not None else []
		t_ids = tracked.tracker_id.tolist() if getattr(tracked, "tracker_id", None) is not None else []
		for i, d in enumerate(dets):
			if track_ids[i] >= 0:
				continue
			d_box = [v for v in xyxy[i]]
			best_j = -1
			best_iou = 0.0
			for j, t_box in enumerate(t_boxes):
				if j in used_tr:
					continue
				iou = _bbox_iou_xyxy(
				    (float(d_box[0]), float(d_box[1]), float(d_box[2]), float(d_box[3])),
				    (float(t_box[0]), float(t_box[1]), float(t_box[2]), float(t_box[3])),
				)
				if iou > best_iou:
					best_iou = iou
					best_j = j
			if best_j >= 0 and best_iou > 0.05 and best_j < len(t_ids):
				track_ids[i] = int(t_ids[best_j])
				used_tr.add(best_j)

	return track_ids


def _assign_team_slots_constrained(
    team_rows: list[dict],
    track_ids: list[int],
    frame_no: int,
    roster_map: dict[int, int],
    roster_last_seen: dict[int, int],
    slot_state: dict[int, dict],
    max_slots: int = 11,
    release_after: int = 20,
) -> list[int]:
	"""
	Assign 1..11 slots with roster constraint, combining:
	- track-based continuity when available
	- nearest-slot fallback by position
	- confidence-prioritized fill for remaining detections
	"""
	n = len(team_rows)
	team_slots = [0] * n
	if n == 0:
		return team_slots

	# Expire very old track->slot bindings.
	expired = [tid for tid, last in roster_last_seen.items() if frame_no - int(last) > release_after]
	for tid in expired:
		roster_last_seen.pop(tid, None)
		roster_map.pop(tid, None)

	used_slots: set[int] = set()

	# Pass 1: keep known track->slot bindings.
	for i, tid in enumerate(track_ids):
		if tid <= 0:
			continue
		slot = int(roster_map.get(tid, 0))
		if 1 <= slot <= max_slots and slot not in used_slots:
			team_slots[i] = slot
			used_slots.add(slot)
			roster_last_seen[tid] = frame_no

	remaining = [i for i, s in enumerate(team_slots) if s <= 0]
	available_slots = [s for s in range(1, max_slots + 1) if s not in used_slots]

	# Pass 2: nearest recent slot by position (one-to-one greedy).
	cands: list[tuple[float, int, int]] = []  # (cost, det_idx, slot)
	for i in remaining:
		cx = float(team_rows[i]["cx"])
		cy = float(team_rows[i]["cy"])
		for s in available_slots:
			state = slot_state.get(s)
			if not state:
				continue
			last = int(state.get("last_frame", -999999))
			age = max(0, frame_no - last)
			if age > release_after:
				continue
			px = float(state.get("cx", 0.0))
			py = float(state.get("cy", 0.0))
			dx = cx - px
			dy = cy - py
			dist = math.sqrt(dx * dx + dy * dy)
			max_dist = min(0.20, 0.06 + 0.01 * age)
			if dist > max_dist:
				continue
			cost = dist + 0.003 * age
			cands.append((cost, i, s))
	cands.sort(key=lambda x: x[0])
	used_det: set[int] = set()
	for _, i, s in cands:
		if i in used_det or s in used_slots:
			continue
		team_slots[i] = s
		used_det.add(i)
		used_slots.add(s)

	# Pass 3: assign remaining detections by confidence to free slots.
	remaining = [i for i, s in enumerate(team_slots) if s <= 0]
	available_slots = [s for s in range(1, max_slots + 1) if s not in used_slots]
	remaining_sorted = sorted(remaining, key=lambda i: float(team_rows[i].get("conf", 0.0)), reverse=True)
	for i, s in zip(remaining_sorted, available_slots):
		team_slots[i] = s
		used_slots.add(s)

	# Pass 4: if detections still remain (e.g., >11 detections for this team),
	# assign nearest existing slot as duplicate so labels are not blank.
	remaining = [i for i, s in enumerate(team_slots) if s <= 0]
	for rank, i in enumerate(remaining):
		cx = float(team_rows[i]["cx"])
		cy = float(team_rows[i]["cy"])
		best_slot = 0
		best_dist = float("inf")
		for s in range(1, max_slots + 1):
			state = slot_state.get(s)
			if not state:
				continue
			px = float(state.get("cx", 0.0))
			py = float(state.get("cy", 0.0))
			dx = cx - px
			dy = cy - py
			d = math.sqrt(dx * dx + dy * dy)
			if d < best_dist:
				best_dist = d
				best_slot = s
		if best_slot <= 0:
			# Early-frame fallback if slot_state is not populated yet.
			best_slot = (rank % max_slots) + 1
		team_slots[i] = best_slot

	# Update track bindings and slot states from current assignments.
	for i, slot in enumerate(team_slots):
		if slot <= 0:
			continue
		tid = int(track_ids[i]) if i < len(track_ids) else -1
		if tid > 0:
			roster_map[tid] = int(slot)
			roster_last_seen[tid] = frame_no
		slot_state[int(slot)] = {
		    "cx": float(team_rows[i]["cx"]),
		    "cy": float(team_rows[i]["cy"]),
		    "last_frame": int(frame_no),
		    "track_id": int(tid),
		}

	return team_slots


def _build_tracking_labels_for_frame(
    players_team_txt: Path,
    players_conf_txt: Path,
    frame_no: int,
    trackers_by_team: dict[int, object],
    roster_map_by_team: dict[int, dict[int, int]],
    roster_seen_by_team: dict[int, dict[int, int]],
    slot_state_by_team: dict[int, dict[int, dict]],
    img_w: int,
    img_h: int,
) -> list[dict]:
	"""
	Build per-player tracking labels with stable team-wise 1..11 IDs.
	Returns list of label rows:
	{player_index, team_id, team_slot, track_id}
	"""
	rows = _load_players_team_rows(players_team_txt)
	conf_by_idx = _load_conf_by_index(players_conf_txt)
	for r in rows:
		# ByteTrack needs confidence; fallback to a small positive value.
		r["conf"] = float(conf_by_idx.get(int(r["idx"]), 0.01))
	labels: list[dict] = []
	for team in (0, 1):
		team_rows = [r for r in rows if int(r.get("team", -1)) == team]
		track_ids = _track_team_with_bytetrack(team_rows, trackers_by_team[team], img_w, img_h)
		team_slots = _assign_team_slots_constrained(
		    team_rows=team_rows,
		    track_ids=track_ids,
		    frame_no=frame_no,
		    roster_map=roster_map_by_team[team],
		    roster_last_seen=roster_seen_by_team[team],
		    slot_state=slot_state_by_team[team],
		    max_slots=11,
		)

		for i, r in enumerate(team_rows):
			slot_i = int(team_slots[i]) if i < len(team_slots) else 0
			labels.append(
			    {
			        "player_index": int(r["idx"]),
			        "team_id": team,
			        "team_slot": slot_i,
			        "track_id": int(track_ids[i]) if i < len(track_ids) else -1,
			    }
			)
	labels.sort(key=lambda x: x["player_index"])
	return labels


def _write_players_label_txt(path: Path, labels: list[dict]) -> None:
	"""
	Write per-player labels used by homography field renderer:
	player_index team_id team_slot track_id
	"""
	lines: list[str] = []
	for item in labels:
		lines.append(
		    f"{int(item['player_index'])} {int(item['team_id'])} "
		    f"{int(item['team_slot'])} {int(item['track_id'])}"
		)
	path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
	print(f"[done] wrote players label txt: {path}")


def _write_players_label_selected_txt(path: Path, labels: list[dict], selected_player_indices: set[int]) -> None:
	"""
	Write labels file where only selected players keep their slot/track;
	others are written with slot=0 and track=-1.
	"""
	out_labels: list[dict] = []
	for item in labels:
		idx = int(item.get("player_index", -1))
		if idx in selected_player_indices:
			out_labels.append(item)
		else:
			out_labels.append(
			    {
			        "player_index": idx,
			        "team_id": int(item.get("team_id", -1)),
			        "team_slot": 0,
			        "track_id": -1,
			    }
			)
	_write_players_label_txt(path, out_labels)


def _parse_pocket_boxes(pocket_txt: Path) -> list[tuple[float, float, float, float]]:
	boxes: list[tuple[float, float, float, float]] = []
	if not pocket_txt.exists():
		return boxes
	for raw in pocket_txt.read_text(encoding="utf-8").splitlines():
		parts = raw.strip().split()
		if len(parts) < 5:
			continue
		try:
			cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
		except Exception:
			continue
		boxes.append((cx, cy, w, h))
	return boxes


def _write_pocket_overlay_image(image_path: Path, pocket_txt: Path, out_path: Path) -> None:
	"""
	Draw pocket detection rectangles from pocket.txt on the original frame image.
	"""
	img = cv2.imread(str(image_path))
	if img is None:
		print(f"[warn] cannot read image for pocket overlay: {image_path}")
		return
	h, w = img.shape[:2]
	boxes = _parse_pocket_boxes(pocket_txt)
	for cx, cy, bw, bh in boxes:
		x1 = int(round((cx - bw / 2.0) * w))
		y1 = int(round((cy - bh / 2.0) * h))
		x2 = int(round((cx + bw / 2.0) * w))
		y2 = int(round((cy + bh / 2.0) * h))
		x1 = max(0, min(w - 1, x1))
		y1 = max(0, min(h - 1, y1))
		x2 = max(0, min(w - 1, x2))
		y2 = max(0, min(h - 1, y2))
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(img, "pocket", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(out_path), img)


def _formation_draw_image(
    out_path: Path,
    players: list[dict],
    yard_x: list[float],
    hash_y: list[float],
    field_width: float = 53.33,
) -> None:
	"""
	Draw first-frame formation image with category colors and labels.
	"""
	if not yard_x:
		return
	canvas_w, canvas_h, pad = 1200, 600, 20
	xs = list(yard_x)
	step = xs[1] - xs[0] if len(xs) > 1 else 5.0
	x_min = xs[0] - step
	x_max = xs[-1] + step
	y_top = 0.0
	y_bottom = field_width
	scale_x = (canvas_w - 2 * pad) / max(1e-6, (x_max - x_min))
	scale_y = (canvas_h - 2 * pad) / max(1e-6, (y_bottom - y_top))

	def to_px(xf: float, yf: float) -> tuple[int, int]:
		px = int(round(pad + (xf - x_min) * scale_x))
		py = int(round(pad + (yf - y_top) * scale_y))
		return px, py

	img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
	img[:, :] = (20, 90, 20)
	for x in [x_min] + xs + [x_max]:
		p1 = to_px(x, y_top); p2 = to_px(x, y_bottom)
		cv2.line(img, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
	for y in hash_y:
		p1 = to_px(x_min, y); p2 = to_px(x_max, y)
		# dotted
		n = 40
		for i in range(n):
			t0 = i / n
			t1 = min((i + 0.5) / n, 1.0)
			x0 = int(round(p1[0] + (p2[0] - p1[0]) * t0))
			y0 = int(round(p1[1] + (p2[1] - p1[1]) * t0))
			x1 = int(round(p1[0] + (p2[0] - p1[0]) * t1))
			y1 = int(round(p1[1] + (p2[1] - p1[1]) * t1))
			cv2.line(img, (x0, y0), (x1, y1), (255, 255, 255), 2, cv2.LINE_AA)

	cat_colors = {
	    # Offense: exactly three category colors
	    "off_wr": (0, 0, 255),          # red
	    "off_lineman": (180, 50, 180),  # purple
	    "off_backs": (0, 170, 0),       # green (darker than cyan)
	    # Defense: keep distinct colors by level
	    "def_dl": (255, 80, 80),        # light red
	    "def_second": (255, 165, 0),    # orange
	    "def_deep": (0, 102, 255),      # blue
	}
	for p in players:
		xf = float(p["x_field"]); yf = float(p["y_field"])
		px, py = to_px(xf, yf)
		cat = str(p.get("level", ""))
		color = cat_colors.get(cat, (0, 255, 255))
		cv2.circle(img, (px, py), 9, color, -1, cv2.LINE_AA)
		label = str(p.get("team_slot", "")) if int(p.get("team_slot", 0)) > 0 else str(p.get("track_id", ""))
		if label and label != "-1":
			# Adaptive text color improves readability on bright fills.
			brightness = (int(color[0]) + int(color[1]) + int(color[2])) / 3.0
			txt_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)
			cv2.putText(img, label, (px - 6, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_color, 1, cv2.LINE_AA)

	out_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(out_path), img)


def _run_first_frame_formation_analysis(
    frame_key: str,
    frame_img: Path,
    txt_dir: Path,
    out_dir: Path,
    selected_rows: list[dict],
) -> None:
	"""
	Analyze first frame only and write formation summary + visualization.
	"""
	formation_dir = out_dir / "formation"
	formation_dir.mkdir(parents=True, exist_ok=True)
	summary_txt = formation_dir / "formation_summary.txt"
	vis_img = formation_dir / f"{frame_key}_formation.jpg"

	if not selected_rows:
		summary_txt.write_text("status=unknown\nreason=no_selected_rows\n", encoding="utf-8")
		return
	if load_inputs is None:
		summary_txt.write_text("status=unknown\nreason=homography_module_unavailable\n", encoding="utf-8")
		return

	# Determine offense side from pocket in image space.
	pocket_boxes = _parse_pocket_boxes(txt_dir / "pocket.txt")
	pocket_cx = pocket_boxes[0][0] if pocket_boxes else None
	team_centers: dict[int, float] = {}
	for team in (0, 1):
		xs = [float(r["cx"]) for r in selected_rows if int(r.get("team", -1)) == team]
		if xs:
			team_centers[team] = float(sum(xs) / len(xs))
	offense_team = None
	if pocket_cx is not None and len(team_centers) == 2:
		offense_team = min(team_centers.keys(), key=lambda t: abs(team_centers[t] - pocket_cx))
	defense_team = 1 - offense_team if offense_team in (0, 1) else None

	data = load_inputs(frame_img, txt_dir)
	hash_sorted, yard_sorted = order_lines_by_intersections(data.harshmark_lines, data.yardline_lines)
	grid = compute_hash_yard_intersections(hash_sorted, yard_sorted)
	field_grid = build_field_grid(len(hash_sorted), len(yard_sorted))
	H, _, _ = compute_homography_from_grid(grid, field_grid)
	if H is None:
		summary_txt.write_text("status=unknown\nreason=homography_failed\n", encoding="utf-8")
		return

	points_img = [(float(r["cx"]), float(r["cy"])) for r in selected_rows]
	points_field = apply_homography_to_points(points_img, H)
	valid_x = [p[0] for p in points_field if p is not None]
	if not valid_x:
		summary_txt.write_text("status=unknown\nreason=no_field_points\n", encoding="utf-8")
		return
	x_los = float(statistics.median(valid_x))

	def_xs = [p[0] for p, r in zip(points_field, selected_rows) if p is not None and int(r.get("team", -1)) == defense_team]
	def_centroid_x = float(sum(def_xs) / len(def_xs)) if def_xs else x_los + 1.0
	sign_def = 1.0 if def_centroid_x >= x_los else -1.0
	sign_off = -sign_def
	y_vals = [p[1] for p in points_field if p is not None]
	y_mid = float(statistics.median(y_vals)) if y_vals else 26.665

	# Use players_labels_selected.txt as the single source of truth for slot labels.
	label_by_idx: dict[int, dict] = {}
	labels_path = txt_dir / "players_labels_selected.txt"
	if labels_path.exists():
		for raw in labels_path.read_text(encoding="utf-8").splitlines():
			parts = raw.strip().split()
			if len(parts) < 4:
				continue
			try:
				p_idx = int(parts[0]); team_id = int(parts[1]); team_slot = int(parts[2]); track_id = int(parts[3])
			except Exception:
				continue
			label_by_idx[p_idx] = {
			    "player_index": p_idx,
			    "team_id": team_id,
			    "team_slot": team_slot,
			    "track_id": track_id,
			}
	player_rows: list[dict] = []
	counts = {
	    "off_wr": 0,
	    "off_lineman": 0,
	    "off_backs": 0,
	    "def_dl": 0,
	    "def_second": 0,
	    "def_deep": 0,
	    "off_in_box": 0,
	    "off_out_box": 0,
	    "def_in_box": 0,
	    "def_out_box": 0,
	}
	raw_rows: list[dict] = []
	for r, pf in zip(selected_rows, points_field):
		if pf is None:
			continue
		idx = int(r["idx"])
		team = int(r.get("team", -1))
		side = "off" if team == offense_team else ("def" if team == defense_team else "unknown")
		depth = (sign_off if side == "off" else sign_def) * (float(pf[0]) - x_los)
		# Explicitly classify pocket membership from original-image pocket box.
		cx_img = float(r.get("cx", 0.0))
		cy_img = float(r.get("cy", 0.0))
		in_pocket = any(
		    (bx - bw / 2.0) <= cx_img <= (bx + bw / 2.0) and
		    (by - bh / 2.0) <= cy_img <= (by + bh / 2.0)
		    for (bx, by, bw, bh) in pocket_boxes
		)
		raw_rows.append(
		    {
		        "idx": idx,
		        "team": team,
		        "side": side,
		        "depth": float(depth),
		        "in_pocket": bool(in_pocket),
		        "x_field": float(pf[0]),
		        "y_field": float(pf[1]),
		        "conf": float(r.get("conf", 0.0)),
		    }
		)

	# Adaptive offense lineman/back threshold (user requested):
	# normalize in-pocket offense depths to [0,1], then threshold = RMS(normalized_depth).
	off_in_pocket_depths = [row["depth"] for row in raw_rows if row["side"] == "off" and row["in_pocket"]]
	off_thresh_norm = 0.5
	if off_in_pocket_depths:
		d_min = min(off_in_pocket_depths)
		d_max = max(off_in_pocket_depths)
		den = max(1e-6, d_max - d_min)
		norms = [(d - d_min) / den for d in off_in_pocket_depths]
		off_thresh_norm = math.sqrt(sum(v * v for v in norms) / max(1, len(norms)))

	for row in raw_rows:
		idx = int(row["idx"])
		side = row["side"]
		depth = float(row["depth"])
		in_pocket = bool(row["in_pocket"])
		if side == "off":
			if not in_pocket:
				level = "off_wr"
			else:
				d_min = min(off_in_pocket_depths) if off_in_pocket_depths else 0.0
				d_max = max(off_in_pocket_depths) if off_in_pocket_depths else 1.0
				den = max(1e-6, d_max - d_min)
				d_norm = (depth - d_min) / den
				level = "off_lineman" if d_norm <= off_thresh_norm else "off_backs"
		elif side == "def":
			if depth <= 1.5:
				level = "def_dl"
			elif depth <= 5.0:
				level = "def_second"
			else:
				level = "def_deep"
		else:
			level = "unknown"

		if side == "off":
			counts["off_in_box" if in_pocket else "off_out_box"] += 1
			if level in ("off_wr", "off_lineman", "off_backs"):
				counts[level] += 1
		elif side == "def":
			counts["def_in_box" if in_pocket else "def_out_box"] += 1
			if level in ("def_dl", "def_second", "def_deep"):
				counts[level] += 1
		lbl = label_by_idx.get(idx, {})
		player_rows.append(
		    {
		        "player_index": int(row["idx"]),
		        "team_id": int(row["team"]),
		        "side": side,
		        "level": level,
		        "box_tag": "in_box" if in_pocket else "out_box",
		        "track_id": int(lbl.get("track_id", -1)),
		        "team_slot": int(lbl.get("team_slot", 0)),
		        "x_field": float(row["x_field"]),
		        "y_field": float(row["y_field"]),
		        "conf": float(row["conf"]),
		    }
		)

	yard_x = [pt[0] for pt in field_grid[0]] if field_grid and field_grid[0] else []
	hash_y = [row[0][1] for row in field_grid if row]
	_formation_draw_image(vis_img, player_rows, yard_x, hash_y)

	lines = [
	    f"frame={frame_key}",
	    f"offense_team={offense_team if offense_team is not None else 'unknown'}",
	    f"defense_team={defense_team if defense_team is not None else 'unknown'}",
	    f"x_los={x_los:.4f}",
	    f"y_mid={y_mid:.4f}",
	    f"off_lineman_backs_threshold_norm={off_thresh_norm:.4f}",
	    "",
	    "[counts]",
	    f"off_wr={counts['off_wr']}",
	    f"off_lineman={counts['off_lineman']}",
	    f"off_backs={counts['off_backs']}",
	    f"def_dl={counts['def_dl']}",
	    f"def_second={counts['def_second']}",
	    f"def_deep={counts['def_deep']}",
	    f"off_in_box={counts['off_in_box']}",
	    f"off_out_box={counts['off_out_box']}",
	    f"def_in_box={counts['def_in_box']}",
	    f"def_out_box={counts['def_out_box']}",
	    "",
	    "[players]",
	    "player_index team_id side level box_tag track_id team_slot x_field y_field conf",
	]
	for p in sorted(player_rows, key=lambda x: x["player_index"]):
		lines.append(
		    f"{p['player_index']} {p['team_id']} {p['side']} {p['level']} {p['box_tag']} "
		    f"{p['track_id']} {p['team_slot']} {p['x_field']:.4f} {p['y_field']:.4f} {p['conf']:.4f}"
		)
	summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"[done] wrote formation summary: {summary_txt}")
	print(f"[done] wrote formation visualization: {vis_img}")


def _build_labels_from_selected_rows(
    selected_rows: list[dict],
    frame_no: int,
    roster_map_by_team: dict[int, dict[int, int]],
    roster_seen_by_team: dict[int, dict[int, int]],
    slot_state_by_team: dict[int, dict[int, dict]],
) -> list[dict]:
	"""
	Build players_labels from the selected 22-player stream only.
	Uses track_id when present for slot continuity.
	"""
	labels: list[dict] = []
	for team in (0, 1):
		team_rows = [r for r in selected_rows if int(r.get("team", -1)) == team]
		track_ids = [int(r.get("track_id", -1)) for r in team_rows]
		team_slots = _assign_team_slots_constrained(
		    team_rows=team_rows,
		    track_ids=track_ids,
		    frame_no=frame_no,
		    roster_map=roster_map_by_team[team],
		    roster_last_seen=roster_seen_by_team[team],
		    slot_state=slot_state_by_team[team],
		    max_slots=11,
		)
		for i, r in enumerate(team_rows):
			labels.append(
			    {
			        "player_index": int(r["idx"]),
			        "team_id": team,
			        "team_slot": int(team_slots[i]) if i < len(team_slots) else 0,
			        "track_id": int(track_ids[i]) if i < len(track_ids) else -1,
			    }
			)
	labels.sort(key=lambda x: x["player_index"])
	return labels


def build_field_video(field_images: list[Path], out_video: Path, fps: float, codec_mode: str = "auto") -> None:
	"""
	Concatenate ordered field images into an MP4 video.
	"""
	if not field_images:
		print("[warn] No field images found; skip video export.")
		return
	first = cv2.imread(str(field_images[0]))
	if first is None:
		print(f"[warn] Cannot read first field image: {field_images[0]}")
		return
	h, w = first.shape[:2]
	out_video.parent.mkdir(parents=True, exist_ok=True)
	writer = None
	used_codec = None
	if codec_mode == "auto":
		codec_candidates = ("avc1", "mp4v")
	elif codec_mode in ("avc1", "mp4v"):
		codec_candidates = (codec_mode,)
	else:
		raise ValueError(f"Unsupported codec mode: {codec_mode}")
	# Prefer avc1 in auto mode for better compatibility with web-based players.
	for codec in codec_candidates:
		fourcc = cv2.VideoWriter_fourcc(*codec)
		candidate = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
		if candidate.isOpened():
			writer = candidate
			used_codec = codec
			break
		candidate.release()
	if writer is None:
		print(f"[warn] Failed to open video writer with {codec_candidates}: {out_video}")
		return
	written = 0
	try:
		for img_path in field_images:
			frame = cv2.imread(str(img_path))
			if frame is None:
				print(f"[warn] skip unreadable field image: {img_path}")
				continue
			if frame.shape[1] != w or frame.shape[0] != h:
				frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
			writer.write(frame)
			written += 1
	finally:
		writer.release()
	print(f"[done] field video written ({written} frames @ {fps} fps, codec={used_codec}): {out_video}")


def iter_sampled_frames(video_path: Path, target_fps: float):
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video: {video_path}")
	src_fps = float(cap.get(cv2.CAP_PROP_FPS))
	if src_fps <= 0:
		src_fps = target_fps
	interval = 1.0 / target_fps
	next_t = 0.0
	frame_idx = 0
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		t = frame_idx / src_fps
		if t + 1e-9 >= next_t:
			yield frame_idx, frame, src_fps
			next_t += interval
		frame_idx += 1
	cap.release()


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Run full per-frame AutoScout pipeline from a video.")
	ap.add_argument("--video", required=True, help="Input video path.")
	ap.add_argument("--fps", required=True, type=float, help="Target sampling FPS (e.g. 1, 2, 5).")
	ap.add_argument("--out-dir", required=True, help="Output root directory.")
	ap.add_argument(
	    "--config",
	    default=str(DEFAULT_CONFIG),
	    help="Inference config JSON path (weights/conf/device). Default: code/Inference_tgt/para_local.json",
	)
	ap.add_argument("--device", help="Optional inference device override, e.g. 'cpu' or '0'.")
	ap.add_argument("--max-frames", type=int, help="Optional cap on sampled frames for quick testing.")
	ap.add_argument(
	    "--video-codec",
	    choices=["auto", "avc1", "mp4v"],
	    default="auto",
	    help="Codec for stitched field video: auto (avc1->mp4v), avc1 (compatible), mp4v (faster).",
	)
	ap.add_argument("--save-indi", action="store_true", help="Save jersey clustering outputs under logs/indi.")
	ap.add_argument(
	    "--team-colors",
	    action="store_true",
	    help="Enable team-colored dots in field visualization (runs clustering per frame).",
	)
	ap.add_argument(
	    "--save-processed-crops",
	    action="store_true",
	    help="With --save-indi or --team-colors, also save processed crops (green-screened previews).",
	)
	ap.add_argument("--save-vis", action="store_true", help="Save overlay visualizations under logs/vis.")
	ap.add_argument(
	    "--save-intersections",
	    action="store_true",
	    help="Save homography intersection images under logs/intersections.",
	)
	ap.add_argument(
	    "--save-team-counts",
	    action="store_true",
	    help="Save logs/team_counts.txt with pocket-priority per-frame Team 0/1 counts.",
	)
	ap.add_argument(
	    "--formation-analysis",
	    action=argparse.BooleanOptionalAction,
	    default=True,
	    help="Run first-frame formation analysis and write outputs to <out-dir>/formation (default: on).",
	)
	return ap.parse_args()


def main() -> None:
	args = parse_args()

	video_path = Path(args.video).expanduser().resolve()
	out_dir = Path(args.out_dir).expanduser().resolve()
	config_path = Path(args.config).expanduser().resolve()

	if not video_path.exists():
		raise SystemExit(f"--video not found: {video_path}")
	if not config_path.exists():
		raise SystemExit(f"--config not found: {config_path}")
	if args.fps <= 0:
		raise SystemExit("--fps must be > 0")

	frames_root = out_dir / "frames"
	logs_root = out_dir / "logs"
	frames_root.mkdir(parents=True, exist_ok=True)
	logs_root.mkdir(parents=True, exist_ok=True)
	tmp_team_root = logs_root / "_tmp_team"
	run_team_clustering = args.save_indi or args.team_colors or args.formation_analysis
	field_images: list[Path] = []
	ref_cluster_means: dict[int, list[float]] | None = None
	team_count_lines: list[str] = []
	if run_team_clustering and sv is None:
		raise SystemExit("ByteTrack dependency missing. Install with: pip install supervision")
	global_tracker = None
	if run_team_clustering and sv is not None:
		frame_rate = int(max(1, round(float(args.fps))))
		global_tracker = sv.ByteTrack(frame_rate=frame_rate)
	roster_map_by_team: dict[int, dict[int, int]] = {0: {}, 1: {}}
	roster_seen_by_team: dict[int, dict[int, int]] = {0: {}, 1: {}}
	slot_state_by_team: dict[int, dict[int, dict]] = {0: {}, 1: {}}
	track_team_score: dict[int, float] = {}
	formation_done = False

	processed = 0
	for src_frame_idx, frame, src_fps in iter_sampled_frames(video_path, args.fps):
		frame_key = f"frame_{src_frame_idx:06d}"
		frame_dir = frames_root / frame_key
		frame_dir.mkdir(parents=True, exist_ok=True)
		frame_img = frame_dir / f"{frame_key}.jpg"

		if not cv2.imwrite(str(frame_img), frame):
			print(f"[warn] failed to write frame image: {frame_img}")
			continue

		# 1) Inference -> official txt outputs in frames/<frame_key>/txt
		infer_cmd = [
		    sys.executable,
		    str(INFER_SCRIPT),
		    "--image",
		    str(frame_img),
		    "--config",
		    str(config_path),
		    "--out-dir",
		    str(frames_root),
		    "--out-name",
		    frame_key,
		]
		if args.device:
			infer_cmd.extend(["--device", args.device])
		run(infer_cmd)

		txt_dir = frame_dir / "txt"
		players_txt = txt_dir / "players.txt"
		pocket_txt = txt_dir / "pocket.txt"

		# Always save pocket debug visualization for each frame.
		pocket_vis_path = logs_root / "pocket_vis" / f"{frame_key}_pocket.jpg"
		_write_pocket_overlay_image(frame_img, pocket_txt, pocket_vis_path)

		# 2) Optional color clustering:
		# - logs/indi/<frame_key> when saving logs
		# - temporary logs/_tmp_team/<frame_key> when only team colors are needed
		team_json_path = None
		if run_team_clustering:
			indi_dir = (logs_root / "indi" / frame_key) if args.save_indi else (tmp_team_root / frame_key)
			cluster_cmd = [
			    sys.executable,
			    str(CLUSTER_SCRIPT),
			    "--image",
			    str(frame_img),
			    "--boxes",
			    str(players_txt),
			    "--out-dir",
			    str(indi_dir),
			]
			if args.save_processed_crops:
				cluster_cmd.append("--save-processed-crops")
			run(cluster_cmd)
			candidate = indi_dir / "team_assignments.json"
			if candidate.exists():
				team_json_path = candidate
				payload = _load_team_json(candidate)
				cur_means = _extract_cluster_means(payload)
				if cur_means:
					if ref_cluster_means is None and len(cur_means) >= 2:
						ref_cluster_means = dict(cur_means)
						print("[info] set reference cluster means from first valid frame")
					elif ref_cluster_means is not None:
						mapping = _choose_cluster_mapping(cur_means, ref_cluster_means)
						_rewrite_team_json_with_mapping(candidate, mapping)
				players_team_txt = txt_dir / "players_team.txt"
				_write_players_team_txt(players_txt, candidate, players_team_txt)
				conf_rows = _load_players_conf_rows(txt_dir / "players_conf.txt")
				img_h, img_w = frame.shape[:2]
				det_track_map = _track_all_with_bytetrack(conf_rows, global_tracker, img_w, img_h) if global_tracker else {}
				raw_team_rows = _load_players_team_rows(players_team_txt)
				stable_rows = _stabilize_team_rows_by_track(
				    players_team_rows=raw_team_rows,
				    conf_rows=conf_rows,
				    det_track_map=det_track_map,
				    track_team_score=track_team_score,
				)
				players_team_tracked_txt = txt_dir / "players_team_tracked.txt"
				_write_players_team_rows(players_team_tracked_txt, stable_rows)
				players_label_txt = txt_dir / "players_labels.txt"
				labels_all = _build_labels_from_selected_rows(
				    selected_rows=stable_rows,
				    frame_no=processed,
				    roster_map_by_team=roster_map_by_team,
				    roster_seen_by_team=roster_seen_by_team,
				    slot_state_by_team=slot_state_by_team,
				)
				_write_players_label_txt(players_label_txt, labels_all)
				stats = _select_players_with_pocket_priority(
				    players_team_txt=players_team_tracked_txt,
				    players_conf_txt=txt_dir / "players_conf.txt",
				    pocket_txt=txt_dir / "pocket.txt",
				    target_per_team=11,
				)
				_write_players_team_selected_txt(
				    txt_dir / "players_team_selected.txt",
				    stats.get("selected_rows", []),
				)
				labels_selected = _build_labels_from_selected_rows(
				    selected_rows=stats.get("selected_rows", []),
				    frame_no=processed,
				    roster_map_by_team=roster_map_by_team,
				    roster_seen_by_team=roster_seen_by_team,
				    slot_state_by_team=slot_state_by_team,
				)
				selected_indices = {int(r.get("idx", -1)) for r in stats.get("selected_rows", []) if int(r.get("idx", -1)) >= 0}
				players_label_selected_txt = txt_dir / "players_labels_selected.txt"
				_write_players_label_selected_txt(players_label_selected_txt, labels_selected, selected_indices)
				if args.formation_analysis and not formation_done:
					_run_first_frame_formation_analysis(
					    frame_key=frame_key,
					    frame_img=frame_img,
					    txt_dir=txt_dir,
					    out_dir=out_dir,
					    selected_rows=stats.get("selected_rows", []),
					)
					formation_done = True
				if args.save_team_counts:
					team_count_lines.append(
					    f"{frame_key} "
					    f"raw_total={stats['raw_total']} raw_team0={stats['raw_team0']} raw_team1={stats['raw_team1']} raw_unknown={stats['raw_unknown']} "
					    f"selected_total={stats['selected_total']} selected_team0={stats['selected_team0']} selected_team1={stats['selected_team1']} "
					    f"target={stats['target_per_team']}"
					)

		# 3) Homography -> official field output in frame dir.
		# We first write intersections to frame_dir so field image is also saved in frame_dir.
		intersections_tmp = frame_dir / f"{frame_key}_intersections.jpg"
		hom_cmd = [
		    sys.executable,
		    str(HOM_SCRIPT),
		    "--image",
		    str(frame_img),
		    "--txt-dir",
		    str(txt_dir),
		    "--out",
		    str(intersections_tmp),
		]
		if team_json_path is not None:
			hom_cmd.extend(["--team-json", str(team_json_path)])
		players_label_selected_txt = txt_dir / "players_labels_selected.txt"
		players_label_txt = txt_dir / "players_labels.txt"
		if players_label_selected_txt.exists():
			hom_cmd.extend(["--players-label-txt", str(players_label_selected_txt), "--only-labeled"])
		elif players_label_txt.exists():
			hom_cmd.extend(["--players-label-txt", str(players_label_txt)])
		run(hom_cmd)
		field_img = frame_dir / f"{frame_key}_field{frame_img.suffix}"
		if field_img.exists():
			field_images.append(field_img)
		else:
			print(f"[warn] Missing field image for {frame_key}: {field_img}")

		# 4) Optional overlays -> logs/vis/<frame_key>/vis
		if args.save_vis:
			vis_out = logs_root / "vis" / frame_key
			run([
			    sys.executable,
			    str(VIS_SCRIPT),
			    "--txt-dir",
			    str(txt_dir),
			    "--image",
			    str(frame_img),
			    "--out-dir",
			    str(vis_out),
			])

		# 5) Intersections are intermediate; keep only when flagged (and store under logs).
		if intersections_tmp.exists():
			if args.save_intersections:
				int_dir = logs_root / "intersections"
				int_dir.mkdir(parents=True, exist_ok=True)
				dest = int_dir / intersections_tmp.name
				shutil.move(str(intersections_tmp), str(dest))
			else:
				intersections_tmp.unlink(missing_ok=True)

		processed += 1
		print(
		    f"[done] frame {processed}: src_idx={src_frame_idx} "
		    f"(src_fps={src_fps:.3f}) -> {frame_dir}"
		)
		if args.max_frames is not None and processed >= args.max_frames:
			print(f"[stop] reached --max-frames={args.max_frames}")
			break

	# Remove temporary clustering outputs when they were only needed for field colors.
	if run_team_clustering and not args.save_indi and tmp_team_root.exists():
		shutil.rmtree(tmp_team_root, ignore_errors=True)

	if args.save_team_counts:
		counts_out = logs_root / "team_counts.txt"
		header = "# frame_key raw_total raw_team0 raw_team1 raw_unknown selected_total selected_team0 selected_team1 target\n"
		content = "\n".join(team_count_lines)
		counts_out.write_text(header + content + ("\n" if content else ""), encoding="utf-8")
		print(f"[done] wrote team counts log: {counts_out}")

	video_name = f"{video_path.stem}_field_fps{args.fps:g}.mp4"
	out_video = out_dir / "videos" / video_name
	build_field_video(field_images, out_video, fps=args.fps, codec_mode=args.video_codec)

	print(f"[summary] processed sampled frames: {processed}")
	print(f"[summary] output root: {out_dir}")


if __name__ == "__main__":
	main()

