"""
pipeline.py - all five modules chained together on one video.

    frame --> A (segment) --> D (Kalman track) --> B (px to cm) --> C (belt speed) --> E (type)

    python pipeline.py --video assets/video/sd_conveyor_2.mp4
    python pipeline.py --px-per-cm 20.5
    python pipeline.py --calib moduleB/calibration.json --cam-height-cm 100 --box-height-cm 10
    python pipeline.py --impl alt          # use the alternative functions

Where the SCALE comes from (Module B), first one that is given wins:
    1. --px-per-cm             measured on the belt, or taken from a reference box
    2. --calib + --cam-height-cm   px/cm = f / Z, with f read from calibration.json
    --box-height-cm: the TOP face of a box is closer to the camera than the belt, so it
    looks bigger. Sizes are corrected with
        px/cm(top) = px/cm(belt) * Z_belt / (Z_belt - box_height)
    With none of these the sizes are printed in pixels and the output says so.

Output: assets/pipeline_output.mp4 plus a table of ID, size, speed and type.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import belt_roi                                                     # noqa: E402
import box_detect                                                   # noqa: E402
import impl                                                         # noqa: E402
from moduleC.optical_flow import estimate_belt_motion               # noqa: E402  (Module C)
from moduleD.kalman_tracker import MultiTracker                     # noqa: E402  (Module D)
from moduleE.recognize import (load_crops, collect_crops, TypeBySize,   # noqa: E402  (Module E)
                               crop_box, TYPE_NAMES)


def get_px_per_cm(args, video_w):
    """Module B: pixels-per-cm on the belt plane, or None if we have no scale."""
    if args.px_per_cm:
        return args.px_per_cm
    if args.calib and args.cam_height_cm:
        with open(args.calib) as f:
            c = json.load(f)
        # Scale focal length if the video is not as wide as the calibration photos.
        fx = c["K"][0][0] * video_w / c["image_size"]["width"]
        return fx / args.cam_height_cm
    return None


def load_recognizer(video, roi):
    """Module E: use the saved crops if we have them, otherwise cut new ones."""
    data = load_crops()
    if data is None:
        data = collect_crops(video, roi=roi)
    return TypeBySize().build(*data)


def main():
    p = argparse.ArgumentParser(description="Full conveyor inspection pipeline.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--px-per-cm", type=float, default=0.0, help="pixels per cm on the belt")
    p.add_argument("--calib", default="", help="calibration.json from module B")
    p.add_argument("--cam-height-cm", type=float, default=0.0, help="camera height above the belt")
    p.add_argument("--box-height-cm", type=float, default=0.0, help="box height, for top-face correction")
    p.add_argument("--n-frames", type=int, default=0, help="0 = the whole video")
    p.add_argument("--show", action="store_true", help="show a live window")
    p.add_argument("--min-frames", type=int, default=8, help="ignore tracks seen fewer times")
    p.add_argument("--no-static", action="store_true",
                   help="keep detections that match the still background (wall panels, tiles)")
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    video = os.path.join(ROOT, a.video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"[!] Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = a.n_frames or total

    roi = belt_roi.get_roi(video)
    ppc = get_px_per_cm(a, W)                     # px per cm on the belt
    ppc_top = None
    if ppc and a.box_height_cm and a.cam_height_cm:
        # The top face is closer, so it needs a bigger px/cm.
        ppc_top = ppc * a.cam_height_cm / max(a.cam_height_cm - a.box_height_cm, 1e-6)
    ppc_box = ppc_top or ppc
    recognizer = load_recognizer(video, roi)

    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    out_path = os.path.join(ROOT, "assets", "pipeline_output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracker = MultiTracker(max_dist=0.12 * W, max_missed=8, min_hits=3)
    static = None if a.no_static else box_detect.BackgroundFilter()
    belt_dx = []                                  # Module C: one dx per frame pair
    prev_gray = None
    print(f"[pipeline] impl={impl.get_mode()}  {os.path.basename(video)}  {W}x{H}  "
          f"{fps:.1f} fps  ROI={roi}")

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes, vis = box_detect.segment_boxes(frame, roi, static=static)   # Module A
        tracker.step(boxes, i)                                         # Module D

        if prev_gray is not None:                                      # Module C
            m = estimate_belt_motion(prev_gray, gray, roi)
            if m is not None:
                belt_dx.append(m[0])
        prev_gray = gray

        for tr in tracker.confirmed():                                 # Module E + B on clean boxes
            b = tr.box
            if b.get("touches_border"):
                continue                                               # size is not reliable
            tr.types.append(recognizer.predict(crop_box(frame, b)))
            typ = TYPE_NAMES.get(tr.vote_type(), "unknown")
            if ppc_box:
                size_txt = f"{b['rw'] / ppc_box:.1f}x{b['rh'] / ppc_box:.1f}cm"
            else:
                size_txt = f"{b['rw']:.0f}x{b['rh']:.0f}px"
            cv2.polylines(vis, [b["pts"]], True, (0, 255, 0), 2)
            cv2.putText(vis, f"ID{tr.id} {typ} {size_txt}", (b["x"], max(b["y"] - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if belt_dx:
            sp = abs(np.median(belt_dx[-30:])) * fps                    # median of the last 30
            txt = f"belt: {sp / ppc:.1f} cm/s" if ppc else f"belt: {sp:.0f} px/s"
            cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(vis)
        if n_frames and (i + 1) % max(n_frames // 20, 1) == 0:
            print(f"  frame {i + 1:4d}/{n_frames}  ({100 * (i + 1) // n_frames:3d}%)  "
                  f"{len(boxes):2d} boxes  {len(tracker.confirmed()):2d} tracked")
        if a.show:
            cv2.imshow("pipeline (q = quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ---------------- report ----------------
    belt_px_s = abs(float(np.median(belt_dx))) * fps if belt_dx else 0.0
    unit = "cm/s" if ppc else "px/s"
    belt_val = belt_px_s / ppc if ppc else belt_px_s
    print(f"\nBelt speed (Module C, optical flow + RANSAC affine): {belt_val:.1f} {unit}")
    print("\nTracked boxes:")
    print(f"  {'ID':>3} {'frames':>7} {'size':>16} {'speed':>12} {'type':>8}")
    for tr in sorted(tracker.all_tracks(), key=lambda t: t.id):
        sz = tr.median_size()
        if tr.hits < a.min_frames or sz is None:
            continue                                             # noise / never fully visible
        size = f"{sz[0] / ppc_box:.1f}x{sz[1] / ppc_box:.1f} cm" if ppc_box else f"{sz[0]:.0f}x{sz[1]:.0f} px"
        v = abs(float(np.median(tr.vx))) * fps if tr.vx else 0.0        # Kalman velocity
        speed = f"{v / ppc:.1f} cm/s" if ppc else f"{v:.0f} px/s"
        typ = TYPE_NAMES.get(tr.vote_type(), "unknown")
        print(f"  {tr.id:>3} {tr.hits:>7} {size:>16} {speed:>12} {typ:>8}")
    if not ppc:
        print("\nNOTE: no scale given -> pixels. Use --px-per-cm or --calib + --cam-height-cm for cm.")
    if static is not None:
        print(f"[pipeline] background filter threw away {static.dropped} detections that never moved")
    print(f"[+] annotated video: {out_path}")


if __name__ == "__main__":
    main()
