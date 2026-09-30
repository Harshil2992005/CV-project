"""
live_demo.py - watch the whole pipeline react to the video in real time.

On screen: the rotated box, the track ID (Kalman, Module D), the type (Module E),
the size (Module B) and the belt speed (Module C).

Keys: q quit | space pause | s save a snapshot
    python live_demo.py --video assets/video/sd_conveyor_2.mp4 [--px-per-cm 15]
    python live_demo.py --headless 120        (no window: process 120 frames, save a snapshot)
    python live_demo.py --impl alt           (use the alternative functions)
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import belt_roi                                                     # noqa: E402
import box_detect                                                   # noqa: E402
import impl                                                         # noqa: E402
from moduleC.optical_flow import estimate_belt_motion               # noqa: E402
from moduleD.kalman_tracker import MultiTracker                     # noqa: E402
from moduleE.recognize import (load_crops, collect_crops, TypeBySize,   # noqa: E402
                               crop_box, TYPE_NAMES)


def main():
    p = argparse.ArgumentParser(description="Live view of the full pipeline.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--px-per-cm", type=float, default=0.0, help="pixels per cm on the belt")
    p.add_argument("--headless", type=int, default=0, help="process N frames without a window")
    p.add_argument("--no-static", action="store_true",
                   help="keep detections that match the still background")
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    video = os.path.join(ROOT, a.video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"[!] Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    roi = belt_roi.get_roi(video)
    data = load_crops() or collect_crops(video, roi=roi)
    recognizer = TypeBySize().build(*data)

    tracker = MultiTracker(max_dist=0.12 * W, max_missed=8, min_hits=3)
    static = None if a.no_static else box_detect.BackgroundFilter()
    scale = min(1.0, 900.0 / H)                 # shrink big videos so they fit the screen
    belt_dx, prev_gray, paused, vis, fi = [], None, False, None, 0
    win = "LIVE pipeline (q=quit space=pause s=save)"
    print(f"[live] impl={impl.get_mode()}")
    print("[+] Live view. q = quit, space = pause, s = save frame")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                # Video finished -> start again with a fresh tracker.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker = MultiTracker(max_dist=0.12 * W, max_missed=8, min_hits=3)
                prev_gray, belt_dx = None, []
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, vis = box_detect.segment_boxes(frame, roi, static=static)
            tracker.step(boxes, fi)
            fi += 1
            if prev_gray is not None:
                m = estimate_belt_motion(prev_gray, gray, roi)
                if m is not None:
                    belt_dx.append(m[0])
            prev_gray = gray
            for tr in tracker.confirmed():
                b = tr.box
                if b.get("touches_border"):
                    continue
                tr.types.append(recognizer.predict(crop_box(frame, b)))
                typ = TYPE_NAMES.get(tr.vote_type(), "unknown")
                size = (f"{b['rw'] / a.px_per_cm:.1f}x{b['rh'] / a.px_per_cm:.1f} cm" if a.px_per_cm
                        else f"{b['rw']:.0f}x{b['rh']:.0f} px")
                cv2.polylines(vis, [b["pts"]], True, (0, 255, 0), 2)
                cv2.putText(vis, f"ID {tr.id} | {typ} | {size}", (b["x"], max(b["y"] - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if vis is None:
            continue
        hud = vis.copy()
        if belt_dx:
            sp = abs(np.median(belt_dx[-30:])) * fps
            cv2.putText(hud, f"belt: {sp / a.px_per_cm:.1f} cm/s" if a.px_per_cm else f"belt: {sp:.0f} px/s",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        show = cv2.resize(hud, (int(W * scale), int(H * scale)))

        if a.headless:
            if fi >= a.headless:
                path = os.path.join(ROOT, "assets", "live_snapshot.png")
                cv2.imwrite(path, show)
                print(f"[+] headless done, snapshot -> {path}")
                break
            continue
        cv2.imshow(win, show)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            path = os.path.join(ROOT, "assets", "live_snapshot.png")
            cv2.imwrite(path, show)
            print(f"[+] snapshot -> {path}")
    cap.release()
    cv2.destroyAllWindows()
    if static is not None:
        print(f"[live] background filter threw away {static.dropped} detections that never moved")


if __name__ == "__main__":
    main()
