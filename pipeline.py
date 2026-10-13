"""
pipeline.py - all five modules chained together on one video (v3: metric sizes per box).

    frame --> A (segment) --> D (Kalman track) --> B (px to cm, PER BOX) --> C (belt speed) --> E (type)

Typical use on a FIXED-camera clip (best accuracy, see scene_check.py):

    python scene_check.py --video assets/video/my.mp4            # is the clip good enough?
    python pipeline.py --video assets/video/my.mp4 --ref-auto --ref-box-cm 30 20 15 \
                       --calib moduleB/calibration.json           # YOUR camera's calibration
    python pipeline.py --video ... --ref-box-cm 30 20 15 --ref-id 3   # ref = track 3 of a first run
    python pipeline.py --video ... --px-per-cm 20.5               # old single scale (rough)

Scale, best first (Module B, see size_model.py):
    --ref-box-cm W D [H] + K     Camera3D: per-box depth AND per-box height (top-face metrology)
    --ref-box-cm W D             PlaneScale: exact for boxes as tall as the reference
    --px-per-cm                  one number for the whole frame (approximate, only a flat belt)
    nothing                      sizes stay in pixels and the output says so.
K comes from --calib (moduleB/calibration.json made with YOUR camera) or --focal-px.

Detection mode (Module A): FIXED camera -> background subtraction against the empty-scene
picture ("plate"), otherwise per-frame belt-colour mode. --plate / --no-plate force it.

Output: assets/pipeline_output.mp4 plus a table of ID, size +/- spread, height, speed, type.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import belt_roi                                                     # noqa: E402
import box_detect                                                   # noqa: E402
import impl                                                         # noqa: E402
import scene_check                                                  # noqa: E402
import size_model                                                   # noqa: E402
from moduleC.optical_flow import estimate_belt_motion               # noqa: E402  (Module C)
from moduleD.kalman_tracker import MultiTracker                     # noqa: E402  (Module D)
from moduleE.recognize import (load_crops, collect_crops, TypeBySize,   # noqa: E402  (Module E)
                               crop_box, TYPE_NAMES)

GOOD_FIT = 0.85          # a measurement frame must have a box-shaped silhouette


def load_recognizer(video, roi):
    """Module E: use the saved crops if we have them, otherwise cut new ones."""
    data = load_crops()
    if data is None:
        data = collect_crops(video, roi=roi)
    return TypeBySize().build(*data)


def good_measure(b):
    """Can this detection be used for a size? (not cut off, box shaped, clear top face)"""
    return (not b.get("touches_border") and b.get("top_quad") is not None and b["fit"] >= 0.7)


def isolated(b, others, gap):
    """True when no other detection is within `gap` px of this box (a touching neighbour would
    merge into its outline and spoil the reference)."""
    for o in others:
        if o is b:
            continue
        dx = max(o["x"] - (b["x"] + b["w"]), b["x"] - (o["x"] + o["w"]), 0)
        dy = max(o["y"] - (b["y"] + b["h"]), b["y"] - (o["y"] + o["h"]), 0)
        if max(dx, dy) < gap:
            return False
    return True


def choose_reference(meas, per_frame, args, W, H):
    """Pick the reference track. Returns (track_id, frame_idx, box) or None.

    The reference must travel ALONE: only frames where no other box is nearby are used."""
    def clean(t):
        return [(f, b) for f, b in meas.get(t, [])
                if good_measure(b) and isolated(b, [bb for _, bb in per_frame[f]], 0.04 * W)]
    if args.ref_id:
        rows = clean(args.ref_id)
    else:
        # --ref-auto: the first box through the scene that has enough clean, isolated frames
        best = None
        for t in sorted(meas, key=lambda t: meas[t][0][0]):
            rows = clean(t)
            if len(rows) >= 5:
                best = (t, rows)
                break
        if best is None:
            return None
        args.ref_id, rows = best
    if not rows:
        return None
    # the frame where the box is nearest to the picture centre = least lens/perspective extremes
    f, b = min(rows, key=lambda r: abs(r[1]["cx"] - W / 2) + abs(r[1]["cy"] - H / 2))
    return args.ref_id, f, b


def build_scale(args, ref_box, W, H, ref_quads=None):
    """PlaneScale / Camera3D from the reference box. Returns (kind, model) or (None, None)."""
    if ref_box is None or not args.ref_box_cm:
        return None, None
    w, d = args.ref_box_cm[0], args.ref_box_cm[1]
    h = args.ref_box_cm[2] if len(args.ref_box_cm) > 2 else 0.0
    K = None
    if args.calib and os.path.exists(args.calib):
        K = size_model.load_K(args.calib, W)
    elif args.focal_px:
        K = np.array([[args.focal_px, 0, W / 2], [0, args.focal_px, H / 2], [0, 0, 1]], np.float64)
    if K is not None and h > 0:
        quads = ref_quads if ref_quads else [ref_box["top_quad"]]
        cam = size_model.Camera3D.from_reference_frames(K, quads, w, d, h)
        if cam is not None:
            return "camera3d", cam
        print("[!] 3D pose from the reference failed (view too flat?) -> using PlaneScale")
    return "plane", size_model.PlaneScale.from_reference(ref_box["top_quad"], w, d, h, args.cam_height_cm)


def measure_box(kind, model, b):
    """(L, S, H) in cm for one detection, or None."""
    if kind == "camera3d":
        return model.measure(b["top_quad"], b["pts"])
    if kind == "plane":
        return model.measure(b["top_quad"])
    return None


def track_speed_cm(kind, model, rows, fps):
    """Belt speed seen by one box, cm/s, from the world position of its top-face centre."""
    pts = []
    for f, b, m in rows:
        c = np.asarray(b["top_quad"], np.float32).mean(0)
        if kind == "camera3d" and m:
            xy = model.ray_plane(c, m[2])[0][:2]
        elif kind == "plane":
            xy = model.to_cm(c)[0]
        else:
            return None
        pts.append((f / fps, xy))
    if len(pts) < 6:
        return None
    t = np.array([p[0] for p in pts])
    xy = np.array([p[1] for p in pts])
    # speed along the dominant direction of motion
    d = xy[-1] - xy[0]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return 0.0
    s = (xy - xy[0]) @ (d / n)
    return float(abs(np.polyfit(t, s, 1)[0]))


def main():
    p = argparse.ArgumentParser(description="Full conveyor inspection pipeline.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--px-per-cm", type=float, default=0.0, help="ONE pixels-per-cm for the whole frame (rough)")
    p.add_argument("--ref-box-cm", type=float, nargs="+", metavar="CM",
                   help="known box: W D [H] in cm (H needed for the 3D mode)")
    p.add_argument("--ref-auto", action="store_true", help="the first clean box in the video is the reference")
    p.add_argument("--ref-id", type=int, default=0, help="track id (from a first run) of the reference box")
    p.add_argument("--calib", default="", help="calibration.json of YOUR camera (module B)")
    p.add_argument("--focal-px", type=float, default=0.0, help="focal length in px if there is no calibration.json")
    p.add_argument("--cam-height-cm", type=float, default=0.0, help="camera height (PlaneScale height correction)")
    p.add_argument("--box-height-cm", type=float, default=0.0, help="(legacy) height for the single-scale top-face fix")
    p.add_argument("--n-frames", type=int, default=0, help="0 = the whole video")
    p.add_argument("--show", action="store_true", help="show a live window")
    p.add_argument("--min-frames", type=int, default=8, help="ignore tracks seen fewer times")
    p.add_argument("--plate", action="store_true", help="force background-subtraction mode")
    p.add_argument("--no-plate", action="store_true", help="force the per-frame colour mode")
    p.add_argument("--auto-belt", action="store_true",
                   help="moving camera: estimate the belt colour automatically instead of the Otsu split")
    p.add_argument("--no-static", action="store_true",
                   help="keep detections that match the still background (wall panels, tiles)")
    p.add_argument("--json", default="", help="also write the result table to this json file")
    p.add_argument("--out", default=os.path.join("assets", "pipeline_output.mp4"),
                   help="annotated video to write (default overwrites assets/pipeline_output.mp4)")
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    video = a.video if os.path.isabs(a.video) else os.path.join(ROOT, a.video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"[!] Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = a.n_frames or total

    # ---- mode: fixed camera -> plate, otherwise per-frame colour ----
    chk = scene_check.check(video)
    use_plate = a.plate or (chk["camera_fixed"] and not a.no_plate)
    plate = scene_check.build_plate(video) if use_plate else None
    roi = {"x": 0, "y": 0, "w": W, "h": H} if use_plate else belt_roi.get_roi(video)
    if a.auto_belt and not use_plate and "belt_lab" not in roi and "hsv_lo" not in roi:
        roi["belt_lab"] = belt_roi.auto_belt_lab(video)
        print(f"[pipeline] auto belt colour (L,a,b) = {roi['belt_lab']}")
    recognizer = load_recognizer(video, roi)
    mode = "background-subtraction (fixed camera)" if use_plate else "belt-colour (moving camera: sizes approximate)"
    print(f"[pipeline] impl={impl.get_mode()}  {os.path.basename(video)}  {W}x{H}  {fps:.1f} fps  mode: {mode}")
    if not chk["camera_fixed"]:
        print(f"[!] camera moves ({chk['camera']['shift_px']} px, zoom {chk['camera']['scale_change']*100:.1f} %): "
              f"cm sizes are NOT reliable on this clip, see scene_check.py")

    tracker = MultiTracker(max_dist=0.12 * W, max_missed=8, min_hits=3)
    static = None if (a.no_static or use_plate) else box_detect.BackgroundFilter()
    meas = defaultdict(list)                    # track id -> [(frame, box)]
    per_frame = defaultdict(list)               # frame -> [(track id, box)]
    types = defaultdict(list)
    belt_dx, prev_gray = [], None

    # ================= pass 1: detect + track =================
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, _ = box_detect.segment_boxes(frame, roi, static=static, plate=plate,
                                            min_area_frac=0.0015 if use_plate else 0.004)   # Module A
        tracker.step(boxes, i)                                                              # Module D
        for tr in tracker.confirmed():
            if tr.last_frame == i:
                meas[tr.id].append((i, tr.box))
                per_frame[i].append((tr.id, tr.box))
                if not tr.box.get("touches_border"):
                    types[tr.id].append(recognizer.predict(crop_box(frame, tr.box)))       # Module E
        if prev_gray is not None:                                                           # Module C
            m = estimate_belt_motion(prev_gray, gray, roi)
            if m is not None:
                belt_dx.append(m[0])
        prev_gray = gray
        if (i + 1) % max(n_frames // 10, 1) == 0:
            print(f"  pass 1: frame {i + 1:4d}/{n_frames}  ({100 * (i + 1) // n_frames:3d}%)  "
                  f"{len(boxes):2d} boxes  {len(tracker.confirmed()):2d} tracked")
    cap.release()

    # ================= scale from the reference box =================
    kind, model, ref_info = None, None, None
    if a.ref_box_cm and (a.ref_auto or a.ref_id):
        ref_info = choose_reference(meas, per_frame, a, W, H)
        if ref_info is None:
            print("[!] no clean reference box found (need a box with a visible top face, not cut off)")
        else:
            rb = [b for f, b in meas[ref_info[0]]
                  if good_measure(b) and isolated(b, [bb for _, bb in per_frame[f]], 0.04 * W)]
            step = max(len(rb) // 25, 1)
            rb = rb[::step]
            kind, model = build_scale(a, ref_info[2], W, H, [b["top_quad"] for b in rb])
            print(f"[pipeline] reference = track {ref_info[0]} (frame {ref_info[1]}), scale model: {kind}")
    elif a.ref_box_cm:
        print("[!] --ref-box-cm needs --ref-auto or --ref-id N (which box is the known one?)")
    ppc = a.px_per_cm or None

    # ================= per-track measurement =================
    results = {}
    cm_rows = defaultdict(list)
    for tid, rows in meas.items():
        est = size_model.SizeEstimator()
        for f, b in rows:
            if not good_measure(b):
                if kind is None and not b.get("touches_border"):
                    est.add(b["rw"], b["rh"])           # pixel fallback: silhouette quad
                continue
            m = measure_box(kind, model, b)
            if m is not None:
                est.add(*m)
                cm_rows[tid].append((f, b, m))
            elif kind is None:
                q = np.asarray(b["top_quad"], np.float32)
                pl = size_model._edge_dims(size_model._order(q))
                est.add(pl[0], pl[1])
        r = est.result()
        if r is not None:
            results[tid] = r

    # ================= pass 2: annotated video =================
    cap = cv2.VideoCapture(video)
    out_path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    unit_cm = kind is not None or ppc

    def fmt(r):
        if kind is not None:
            return f"{r['L']:.1f}x{r['S']:.1f}cm"
        if ppc:
            return f"{r['L'] / ppc:.1f}x{r['S'] / ppc:.1f}cm"
        return f"{r['L']:.0f}x{r['S']:.0f}px"

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame
        for tid, b in per_frame.get(i, []):
            r = results.get(tid)
            col = (0, 165, 255) if b.get("touches_border") else (0, 255, 0)
            cv2.polylines(vis, [b["pts"]], True, col, 2)
            if b.get("top_quad") is not None:
                cv2.polylines(vis, [np.asarray(b["top_quad"], np.int32)], True, (255, 200, 0), 1)
            typ = TYPE_NAMES.get(max(set(types[tid]), key=types[tid].count), "unknown") if types[tid] else "unknown"
            cv2.putText(vis, f"ID{tid} {typ} {fmt(r) if r else '...'}", (b["x"], max(b["y"] - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        writer.write(vis)
    cap.release()
    writer.release()

    # ================= report =================
    print("\nTracked boxes   (size = median of clean frames +/- std; H = estimated box height):")
    print(f"  {'ID':>3} {'frames':>6} {'size':>20} {'H':>7} {'speed':>11} {'type':>8}  note")
    table = []
    speeds = []
    for tid in sorted(results):
        tr_hits = len(meas[tid])
        if tr_hits < a.min_frames:
            continue
        r = results[tid]
        sp = track_speed_cm(kind, model, cm_rows.get(tid, []), fps) if kind else None
        if sp:
            speeds.append(sp)
        if kind:
            size = f"{r['L']:.1f}x{r['S']:.1f}cm (±{max(r['spread_L'], r['spread_S']):.1f})"
        elif ppc:
            size = f"{r['L'] / ppc:.1f}x{r['S'] / ppc:.1f}cm"
        else:
            size = f"{r['L']:.0f}x{r['S']:.0f}px"
        typ = TYPE_NAMES.get(max(set(types[tid]), key=types[tid].count), "unknown") if types[tid] else "unknown"
        note = []
        if not kind:
            typ += "?"                 # type comes from a pixel size: unreliable without a metric scale (B9)
        if r["n"] < 8:
            note.append("few clean frames")
        if max(r["spread_L"], r["spread_S"]) > 0.08 * r["L"]:
            note.append("unstable size")
        if tid == (ref_info[0] if ref_info else None):
            note.append("REFERENCE")
        hs = f"{r['H']:.0f}cm" if r["H"] else "-"
        spd = f"{sp:.1f}cm/s" if sp else "-"
        print(f"  {tid:>3} {r['n']:>3}/{r['n_total']:<2} {size:>20} {hs:>7} {spd:>11} {typ:>8}  {', '.join(note)}")
        table.append({"id": tid, "frames_used": r["n"], "L": r["L"], "S": r["S"], "H": r["H"],
                      "spread": max(r["spread_L"], r["spread_S"]), "speed_cm_s": sp, "type": typ,
                      "unit": "cm" if (kind or ppc) else "px", "notes": note})
    if speeds:
        print(f"\nBelt speed (median over boxes, metric): {np.median(speeds):.1f} cm/s")
    elif belt_dx:
        sp = abs(float(np.median(belt_dx))) * fps
        print(f"\nBelt speed (Module C, optical flow + RANSAC affine): "
              f"{sp / ppc:.1f} cm/s" if ppc else f"\nBelt speed (Module C): {sp:.0f} px/s (no scale)")
    if not (kind or ppc):
        print("\nNOTE: no scale -> pixels. Give a known box: --ref-box-cm W D H --ref-auto [--calib K.json]")
    if static is not None:
        print(f"[pipeline] background filter threw away {static.dropped} detections that never moved")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"mode": mode, "scale_model": kind, "camera_fixed": chk["camera_fixed"], "boxes": table}, f, indent=2)
    print(f"[+] annotated video: {out_path}")


if __name__ == "__main__":
    main()
