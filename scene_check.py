"""
scene_check.py - is this video good enough for measuring? + the static background plate.

    python scene_check.py --video assets/video/sd_conveyor_2.mp4

It answers the questions that decide whether ANY detector can reach high accuracy
(problems A1, A2 of the review) instead of leaving them as a feeling:

  camera_shift   does the camera move / zoom during the clip? (ORB matches on the static
                 background + RANSAC similarity; the box minority is rejected as outliers)
  sharpness      variance of the Laplacian (motion blur kills edges and corners)
  verdict        FIXED  -> background-subtraction mode (best accuracy)
                 MOVING -> per-frame colour mode only (approximate sizes, say so in reports)

build_plate() makes the median background picture of a FIXED camera. Boxes pass through,
so the median per pixel is the empty belt + room. The detector then looks for "what is
different from the empty scene", which does not care about belt colour, walls or panels.
"""
import argparse
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PLATE_DIR = os.path.join(ROOT, "assets", "plates")


def _sample(video, n, scale=0.5):
    cap = cv2.VideoCapture(video)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, max(tot - 2, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append((int(i), cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)))
    cap.release()
    return out


def camera_shift(video, n=24):
    """Max translation (px, in the ORIGINAL video resolution) and scale change vs the first sample."""
    fr = _sample(video, n, 0.5)
    if len(fr) < 3:
        return {"shift_px": 0.0, "scale_change": 0.0, "n_ok": 0}
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    g0 = cv2.cvtColor(fr[0][1], cv2.COLOR_BGR2GRAY)
    k0, d0 = orb.detectAndCompute(g0, None)
    worst_t, worst_s, ok_n = 0.0, 0.0, 0
    for _, f in fr[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        k, d = orb.detectAndCompute(g, None)
        if d is None or d0 is None or len(k) < 20:
            continue
        m = bf.match(d0, d)
        if len(m) < 20:
            continue
        a = np.float32([k0[x.queryIdx].pt for x in m])
        b = np.float32([k[x.trainIdx].pt for x in m])
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inl is None or inl.sum() < 12:
            continue
        ok_n += 1
        s = float(np.hypot(M[0, 0], M[1, 0]))
        h, w = g.shape
        c = np.array([w / 2, h / 2, 1.0])
        t = float(np.linalg.norm(M @ c - c[:2])) * 2.0          # back to full resolution
        worst_t, worst_s = max(worst_t, t), max(worst_s, abs(s - 1.0))
    if ok_n < 5:
        # Plain scene (too few corners to match). The top fifth of the picture is the room above
        # the belt: if the camera is fixed, it must equal the median picture there in every sample.
        top = np.stack([f[: int(0.2 * f.shape[0])] for _, f in fr]).astype(np.float32)
        med = np.median(top, axis=0)
        frac = float(np.mean([(np.abs(t - med).max(axis=2) > 25).mean() for t in top]))
        return {"shift_px": 0.0 if frac < 0.02 else 99.0, "scale_change": 0.0, "n_ok": 5,
                "method": "residual", "changed_fraction": round(frac, 4)}
    return {"shift_px": round(worst_t, 2), "scale_change": round(worst_s, 4), "n_ok": ok_n, "method": "orb"}


def sharpness(video, n=12):
    """99.5th percentile of |Laplacian| at 640 px width: how crisp the sharpest edges are.

    (The plain Laplacian variance is also tiny for a clean, untextured scene, so it cannot
    tell "blurred" from "simple". The strongest edges can.)
    """
    v = []
    for _, f in _sample(video, n, 1.0):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
        v.append(np.percentile(np.abs(cv2.Laplacian(g, cv2.CV_64F)), 99.5))
    return float(np.median(v)) if v else 0.0


def check(video):
    cam = camera_shift(video)
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    tol = max(3.0, 0.003 * W)
    fixed = cam["n_ok"] >= 5 and cam["shift_px"] <= tol and cam["scale_change"] <= 0.01
    sharp = sharpness(video)
    advice = []
    if not fixed:
        advice.append("Camera moves/zooms: record again with the camera on a tripod/fixed mount. "
                      "Sizes from this clip are approximate only.")
    if sharp < 18:
        advice.append("Frames are soft/blurred (edge sharpness %.0f): raise shutter speed (>=1/500) or "
                      "add light, otherwise box edges cannot be found precisely." % sharp)
    advice.append("Put ONE box of known size (W x D in cm) first on the belt, flat, top face "
                  "visible: --ref-auto uses it as the scale for every other box.")
    return {"video": os.path.basename(video), "camera": cam, "sharpness": round(sharp, 1),
            "camera_fixed": bool(fixed), "mode": "background-subtraction" if fixed else "colour (approximate)",
            "advice": advice}


def plate_path(video):
    return os.path.join(PLATE_DIR, os.path.splitext(os.path.basename(video))[0] + ".png")


def build_plate(video, n=60, force=False):
    """Median background of a FIXED camera, full resolution BGR (cached in assets/plates/)."""
    pp = plate_path(video)
    if os.path.exists(pp) and not force and os.path.getmtime(pp) >= os.path.getmtime(video):
        return cv2.imread(pp)          # cached, and the video has not been replaced since
    cap = cv2.VideoCapture(video)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(tot - 2, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    plate = np.median(np.stack(frames), axis=0).astype(np.uint8)
    os.makedirs(PLATE_DIR, exist_ok=True)
    cv2.imwrite(pp, plate)
    return plate


def plate_mask(frame, plate, thr=14.0):
    """Foreground = LAB distance to the empty scene, with shadows (darker, same chroma) removed."""
    a = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(plate, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = a - b
    dist = np.sqrt(0.6 * d[:, :, 0] ** 2 + d[:, :, 1] ** 2 + d[:, :, 2] ** 2)
    shadow = (np.abs(d[:, :, 1]) < 4) & (np.abs(d[:, :, 2]) < 4) & (d[:, :, 0] < 0) & (d[:, :, 0] > -45)
    m = ((dist > thr) & ~shadow).astype(np.uint8) * 255
    return m


def main():
    p = argparse.ArgumentParser(description="Check if a video is suitable for accurate box measuring.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    v = a.video if os.path.isabs(a.video) else os.path.join(ROOT, a.video)
    r = check(v)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    print(f"video          : {r['video']}")
    print(f"camera shift   : {r['camera']['shift_px']} px, scale change {r['camera']['scale_change']*100:.1f} %"
          f"  (from {r['camera']['n_ok']} matched samples)")
    print(f"sharpness      : {r['sharpness']}")
    print(f"camera fixed   : {r['camera_fixed']}   -> mode: {r['mode']}")
    for t in r["advice"]:
        print("  *", t)


if __name__ == "__main__":
    main()
