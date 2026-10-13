"""
belt_roi.py - find, load and save the conveyor belt area (the "ROI").

What is a ROI?
    Just a rectangle written as {"x", "y", "w", "h"} in normal full-video pixels.
    Every detector first CUTS the frame down to this rectangle and only looks at the
    pixels inside it. We crop instead of blacking out the outside, because blacking
    out used to create fake dark blobs that the detector then called "boxes".

How the ROI is found automatically
    1. Read 40 frame PAIRS spread through the video.
    2. Blur both frames and take the difference. Where something MOVES, the
       difference is big.
    3. Add up the difference for every image ROW (a row = one horizontal line).
       The belt rows get a big total, the walls / floor get a small total.
    4. Split those row totals with Otsu's method, i.e. at the value that makes
       "moving rows" and "still rows" as far apart as possible, then keep only
       the moving rows and add a small margin above and below.
       A fixed 12 % cut-off was tried first, but it is far too low: the walls and
       the floor keep a little noise, so nearly every row passed and the belt
       area grew to the whole picture. That is what made the detector report wall
       panels and floor tiles as boxes.

Safety check
    The saved file remembers WHICH video it belongs to. So an ROI picked for video A
    is never used on video B by accident.

    python belt_roi.py --video assets/video/sd_conveyor_2.mp4 --save
"""
import argparse
import json
import os

import cv2
import numpy as np

import impl

ROOT = os.path.dirname(os.path.abspath(__file__))

# Where the picked ROI is stored.
ROI_FILE = os.path.join(ROOT, "assets", "belt_roi.json")


# ----------------------------------------------------------------------
# alternative helpers
# ----------------------------------------------------------------------

def smooth_rows_numpy(rows, k=9):
    """ORIGINAL way of smoothing: NumPy convolution (a moving average over the rows)."""
    kernel = np.ones(k, np.float64) / k
    return np.convolve(rows, kernel, mode="same")


def smooth_rows_opencv(rows, k=9):
    """ALTERNATIVE way of doing the exact same moving average, but with OpenCV.

    A moving average over one axis is just a box filter. We lay the 1-D row of
    numbers out as a 1 x H image and blur it with a (k x 1) box filter, so only the
    horizontal neighbours are mixed.

    Two details make it give the SAME answer as np.convolve(..., "same"):
      * np.convolve "same" pads the ends with zeros, so we add those zeros ourselves
        with np.pad and then cut them off again.
      * OpenCV repeats the edge pixels by default, so borderType must be set to
        BORDER_CONSTANT.
    """
    pad = k // 2
    padded = np.pad(np.asarray(rows, np.float64), pad, mode="constant")   # zeros at both ends
    strip = padded.reshape(1, -1).astype(np.float32)
    blurred = cv2.blur(strip, (k, 1), borderType=cv2.BORDER_CONSTANT)
    return blurred.ravel().astype(np.float64)[pad:-pad]


def smooth_rows(rows, k=9):
    """Pick the smoothing function that matches the current impl mode."""
    return smooth_rows_opencv(rows, k) if impl.is_alt() else smooth_rows_numpy(rows, k)


def otsu_rows_numpy(rows):
    """ORIGINAL way of splitting the row totals: Otsu's method written out by hand.

    Otsu slides a cut-off through the values and keeps the position that makes the
    two groups (rows that move and rows that do not) differ as much as possible.
    Here the values are the normalised row totals, so the answer comes back in the
    same 0..1 scale. This is the same idea as the Otsu threshold in box_detect.py,
    only applied to a list of numbers instead of to a picture.
    """
    v = np.clip(np.asarray(rows, np.float64), 0.0, 1.0)
    hist, edges = np.histogram(v, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    csum = np.cumsum(p * centers)
    total = csum[-1]
    m0 = csum / np.maximum(w0, 1e-12)
    m1 = (total - csum) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def otsu_rows_skimage(rows):
    """ALTERNATIVE way of splitting the row totals: scikit-image picks the cut-off."""
    from skimage.filters import threshold_otsu
    v = np.asarray(rows, np.float64)
    if v.max() <= v.min():
        return float(v.max())                   # everything is the same, nothing to split
    return float(threshold_otsu(v))


def rows_threshold(rows, min_frac=0.12):
    """The row total that separates the belt from the room, for the current impl mode.

    min_frac stays as a floor so that a video where the belt moves only a little is
    not cut away completely.
    """
    t = otsu_rows_skimage(rows) if impl.is_alt() else otsu_rows_numpy(rows)
    return max(t, min_frac)


def longest_run(flags):
    """True/False list -> (start, end) of the longest run of True values.

    A second belt-like band far away (a moving fan, a person walking past) would
    otherwise stretch the ROI from the top of the picture to the bottom, so only the
    strongest band is kept.
    """
    best = (0, -1)
    start = None
    for i, on in enumerate(list(flags) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


# ----------------------------------------------------------------------
# automatic ROI detection
# ----------------------------------------------------------------------

def auto_detect_roi(video_path, n_pairs=40, min_frac=0.12):
    """Find the belt by looking for the horizontal band where most motion happens."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"[!] Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # One number per image row. It will hold "how much moved in this row".
    rows = np.zeros(H, np.float64)
    scale = 0.5                       # work on a half-size copy, it is 4x faster

    # np.linspace gives evenly spaced sample positions between 0 and n-4.
    for i in np.linspace(0, max(n - 4, 0), n_pairs).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok1, f1 = cap.read()          # frame i
        cap.read()                    # frame i+1  (skipped, just to move forward)
        ok2, f2 = cap.read()          # frame i+2
        if not (ok1 and ok2):
            continue                  # video ended early, ignore this pair

        # Shrink -> grey -> blur. Blurring first hides sensor noise.
        g1 = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f1, None, fx=scale, fy=scale),
                                           cv2.COLOR_BGR2GRAY), (5, 5), 0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f2, None, fx=scale, fy=scale),
                                           cv2.COLOR_BGR2GRAY), (5, 5), 0)

        # absdiff = per pixel |a - b|. Anything that moved shows up bright here.
        d = cv2.absdiff(g1, g2).astype(np.float32)
        d = cv2.resize(d, (W, H))     # back to full size
        rows += d.mean(axis=1)        # add up each row's average difference

    cap.release()

    # A completely still video gives all zeros, then we cannot know anything.
    if rows.max() <= 0:
        return {"x": 0, "y": 0, "w": W, "h": H}

    # Normalise to 0..1, then smooth so one noisy row cannot cut the belt in half.
    rows = smooth_rows(rows / rows.max())

    keep = rows > rows_threshold(rows, min_frac)      # row numbers that are "active enough"
    if not keep.any():
        return {"x": 0, "y": 0, "w": W, "h": H}   # nothing passed, use the whole frame

    # Only the strongest band of moving rows is the belt.
    y0, y1 = longest_run(keep)
    if y1 <= y0:
        return {"x": 0, "y": 0, "w": W, "h": H}   # nothing passed, use the whole frame

    margin = int(0.04 * H)              # margin is relative, so HD and SD both work
    top = max(y0 - margin, 0)
    bottom = min(y1 + margin, H)
    return {"x": 0, "y": top, "w": W, "h": bottom - top}   # belt always spans full width


# ----------------------------------------------------------------------
# save / load
# ----------------------------------------------------------------------

CONFIG_FILE = os.path.join(ROOT, "assets", "video_configs.json")     # one entry PER VIDEO


def _read_configs():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_roi(roi, video_path=None):
    """Write the ROI for THIS video into assets/video_configs.json (other videos are kept).

    The old single belt_roi.json was overwritten by every video, so tuning one clip
    destroyed the settings of the previous one (review point B2).
    """
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    cfg = _read_configs()
    key = os.path.basename(video_path) if video_path else "_default"
    cfg[key] = dict(roi)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[+] Belt ROI for {key} saved -> {CONFIG_FILE}")


def _clean_roi(d):
    roi = {k: int(d[k]) for k in ("x", "y", "w", "h")}
    for k in ("poly", "hsv_lo", "hsv_hi", "belt_lab"):      # optional: belt polygon + cardboard colour range
        if k in d:
            roi[k] = d[k]
    return roi


def load_roi(video_path=None):
    """The saved ROI for THIS video (new registry first, old belt_roi.json as fallback)."""
    key = os.path.basename(video_path) if video_path else None
    cfg = _read_configs()
    if key and key in cfg:
        return _clean_roi(cfg[key])
    if not os.path.exists(ROI_FILE):
        return None
    with open(ROI_FILE) as f:
        d = json.load(f)
    # Guard: a saved ROI from another video is ignored instead of being used wrongly.
    if video_path and d.get("video") and d["video"] != key:
        return None
    return _clean_roi(d)


def get_roi(video_path):
    """The ROI for this video: the saved one if it fits, otherwise detect it now."""
    roi = load_roi(video_path)
    return roi if roi is not None else auto_detect_roi(video_path)


def in_roi(cx, cy, roi):
    """Is this centre point (cx, cy) inside the belt rectangle?"""
    return roi["x"] <= cx <= roi["x"] + roi["w"] and roi["y"] <= cy <= roi["y"] + roi["h"]


def main():
    p = argparse.ArgumentParser(description="Find / save the conveyor belt rectangle.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--save", action="store_true", help="remember the ROI in assets/belt_roi.json")
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    v = os.path.join(ROOT, a.video)
    roi = auto_detect_roi(v)
    print(f"impl mode: {impl.get_mode()}")
    print("Detected belt ROI (x,y,w,h):", roi)
    if a.save:
        save_roi(roi, v)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# automatic belt colour (fixes B2: no more hand-sampled belt_lab per video)
# ----------------------------------------------------------------------

def auto_belt_lab(video_path, n_frames=48, neutral_tol=16.0):
    """Estimate the belt colour (L, a, b) of a video without any hand sampling.

    Idea: a belt is a big, plain, GREY-ish surface that is present in every frame, while
    boxes come and go. So we (1) take a temporal MEDIAN picture of the whole video, which
    removes the passing boxes, (2) look only at pixels that are close to neutral grey in
    chroma (walls, vests and cardboard are not), (3) take the most common chroma there.
    """
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stack = []
    for i in np.linspace(0, max(n - 2, 0), n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            stack.append(cv2.resize(f, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    cap.release()
    if not stack:
        return None
    plate = np.median(np.stack(stack), axis=0).astype(np.uint8)
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    # keep the lower 70 % of the picture: the belt is never at the very top of the frame
    lab = lab[int(0.3 * lab.shape[0]):]
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    chroma = np.hypot(a - 128, b - 128)
    ok = (chroma < neutral_tol) & (L > 35) & (L < 175)
    if ok.sum() < 200:
        ok = (L > 35) & (L < 175)
    aa, bb = a[ok], b[ok]
    H, xe, ye = np.histogram2d(aa, bb, bins=[32, 32], range=[[96, 160], [96, 160]])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    sel = ok & (np.abs(a - (xe[i] + 1)) < 3) & (np.abs(b - (ye[j] + 1)) < 3)
    return [round(float(np.median(L[sel])), 1), round(float(np.median(a[sel])), 1),
            round(float(np.median(b[sel])), 1)]
