"""
Module A - Segmentation. 5 frames x 5 methods -> one comparison picture.

The 5 methods we compare
    1. Watershed      - distance transform seeds + cv2.watershed
    2. Split & merge  - quadtree split, then merge neighbours that look alike (from scratch)
    3. Felzenszwalb   - graph based segmentation from scikit-image
    4. Mean shift     - colour + position clustering from scikit-learn
    5. N-Cut          - normalized cut on a superpixel graph from scikit-image

    python moduleA/segmentation.py --video assets/video/sd_conveyor_2.mp4
    python moduleA/segmentation.py --impl alt
Output: assets/segmentation_grid.png   (one ROW per frame, one COLUMN per method)

Which method is best when two boxes touch, and why?
    Watershed wins. The distance transform makes one peak per box, so two touching
    boxes give two markers and the flooding cuts them apart at the thin "neck" between
    them.
    Split & merge and Felzenszwalb only look at how similar the colours are, so two
    touching boxes with the same colour become ONE region. Mean shift also groups by
    colour, so it fuses same-coloured neighbours too, and it is slow. N-Cut is global
    and heavy, and it can also cut a single box in half.

Alternative functions (see impl.py)
    base : watershed seeds come from box_detect, region borders drawn with
           skimage.segmentation.mark_boundaries
    alt  : watershed seeds come from skimage peak_local_max, region borders drawn with
           plain OpenCV contours (no scikit-image needed for drawing)
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import belt_roi          # noqa: E402
import box_detect        # noqa: E402
import impl              # noqa: E402


# ----------------------------------------------------------------------
# drawing the region borders
# ----------------------------------------------------------------------

def colorize_skimage(labels, base):
    """ORIGINAL drawing: scikit-image draws the region borders straight onto the photo."""
    from skimage.segmentation import mark_boundaries
    out = mark_boundaries(base[..., ::-1], labels.astype(int), color=(1, 1, 0), mode="inner")
    return (out * 255).astype(np.uint8)


def colorize_opencv(labels, base):
    """ALTERNATIVE drawing: find the region borders with plain NumPy and paint them.

    A border is simply a place where two neighbouring pixels have DIFFERENT region
    numbers, so we compare each pixel with its right neighbour and its lower neighbour
    and mark the ones that disagree. No scikit-image needed at all, and no 8-bit
    limit on the number of regions.
    """
    lab = labels.astype(np.int32)
    edge = np.zeros(lab.shape, bool)
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]          # vertical borders
    edge[:-1, :] |= lab[:-1, :] != lab[1:, :]          # horizontal borders
    out = base.copy()
    out[edge] = (0, 255, 255)                          # BGR yellow
    return out


def colorize(labels, base):
    """Show the region map on top of the photo, chosen by the impl mode."""
    return colorize_opencv(labels, base) if impl.is_alt() else colorize_skimage(labels, base)


# ----------------------------------------------------------------------
# 1. watershed
# ----------------------------------------------------------------------

def seg_watershed_from_detector(frame, roi):
    """ORIGINAL seeds: just reuse the box detector and paint each detected box."""
    boxes, _ = box_detect.segment_boxes(frame, roi)
    lab = np.zeros(frame.shape[:2], np.int32)
    for i, b in enumerate(boxes, 1):        # region 0 stays "background"
        cv2.fillPoly(lab, [b["pts"]], i)
    return lab


def seg_watershed_from_peaks(frame, roi):
    """ALTERNATIVE seeds: find the "deep points" of the distance transform ourselves.

    Steps: work only inside the belt -> make a mask of the moving stuff -> distance
    transform (distance to the nearest background pixel) -> pick every local maximum
    with skimage.feature.peak_local_max -> feed those as watershed markers.
    This is the textbook way and it does not need the box detector at all.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    H, W = frame.shape[:2]
    if roi is None:
        roi = {"x": 0, "y": 0, "w": W, "h": H}
    ox, oy = max(int(roi["x"]), 0), max(int(roi["y"]), 0)
    work = frame[oy:oy + int(roi["h"]), ox:ox + int(roi["w"])]
    h, w = work.shape[:2]
    if h < 10 or w < 10:
        return np.zeros((H, W), np.int32)

    lab_img = cv2.cvtColor(cv2.GaussianBlur(work, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    belt = np.median(lab_img.reshape(-1, 3), axis=0)
    d8 = np.clip(np.linalg.norm(lab_img - belt, axis=2) * 4, 0, 255).astype(np.uint8)
    binary = (d8 > box_detect.otsu_threshold(d8)).astype(np.uint8)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return np.zeros((H, W), np.int32)

    # Every blob gets at least one peak, no matter how small.
    coords = peak_local_max(dist, labels=binary, min_distance=12, num_peaks=60)
    markers = np.zeros(binary.shape, np.int32)
    for i, (py, px) in enumerate(coords, 1):
        markers[py, px] = i
    if markers.max() == 0:                       # no peak found, keep one blob only
        markers[binary > 0] = 1

    markers = watershed(-dist, markers, mask=binary)

    # Blow the small crop result back up to the full frame size.
    out = np.zeros((H, W), np.int32)
    out[oy:oy + h, ox:ox + w] = markers.astype(np.int32)
    return out


def seg_watershed(frame, roi):
    """Watershed segmentation, chosen by the impl mode."""
    return seg_watershed_from_peaks(frame, roi) if impl.is_alt() else seg_watershed_from_detector(frame, roi)


# ----------------------------------------------------------------------
# 2. split & merge, written from scratch
# ----------------------------------------------------------------------

def split_merge(gray, split_var=60.0, merge_thr=12.0, min_size=8):
    """Quadtree split on brightness variance, then merge neighbours with similar means."""
    # Round the size up to a power of two so the quadtree can split evenly.
    n = 1
    while n < max(gray.shape):
        n *= 2
    g = cv2.resize(gray, (n, n)).astype(np.float32)
    lab = np.zeros((n, n), np.int32)
    leaves = []                                     # one entry per final square: (y, x, size)

    def split(y, x, s):
        """Recursively cut a square in 4 while it is still 'busy' enough to split."""
        blk = g[y:y + s, x:x + s]
        # Stop when the square is small, or when it is flat enough (low variance).
        if s <= min_size or blk.var() <= split_var:
            leaves.append((y, x, s))
            return
        h = s // 2
        for dy in (0, h):
            for dx in (0, h):
                split(y + dy, x + dx, h)

    split(0, 0, n)

    # Give every leaf a number, and remember its average brightness and its size.
    mean, cnt = [], []
    for i, (y, x, s) in enumerate(leaves):
        lab[y:y + s, x:x + s] = i
        mean.append(g[y:y + s, x:x + s].mean())
        cnt.append(s * s)
    mean, cnt = np.array(mean), np.array(cnt, float)

    # Union-Find structure: find() says which group a leaf currently belongs to.
    parent = list(range(len(leaves)))

    def find(a):
        """Walk up the parent chain until we reach the group root (path halving)."""
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Look at the label picture and collect every pair of labels that touch.
    pairs = set()
    for a_, b_ in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = a_ != b_
        pairs |= set(zip(a_[m].tolist(), b_[m].tolist()))

    # Merge the most similar neighbours first, so the best joins happen early.
    for a_, b_ in sorted(pairs, key=lambda p: abs(mean[p[0]] - mean[p[1]])):
        ra, rb = find(a_), find(b_)
        if ra != rb and abs(mean[ra] - mean[rb]) < merge_thr:
            tot = cnt[ra] + cnt[rb]
            mean[ra] = (mean[ra] * cnt[ra] + mean[rb] * cnt[rb]) / tot   # weighted mean
            cnt[ra] = tot
            parent[rb] = ra                                                # join the groups

    out = np.array([find(i) for i in range(len(leaves))])[lab]
    return cv2.resize(out.astype(np.int32), (gray.shape[1], gray.shape[0]),
                      interpolation=cv2.INTER_NEAREST)   # back to the real frame size


def seg_split_merge(frame, roi):
    """Split & merge on a slightly blurred grey picture."""
    return split_merge(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0))


# ----------------------------------------------------------------------
# 3. Felzenszwalb
# ----------------------------------------------------------------------

def seg_felzenszwalb(frame, roi):
    """Graph based segmentation from scikit-image. Big frames are shrunk first."""
    from skimage.segmentation import felzenszwalb
    sc = 0.5 if frame.shape[1] > 900 else 1.0       # work at half size on HD video
    small = cv2.resize(frame, None, fx=sc, fy=sc)
    lab = felzenszwalb(small[..., ::-1], scale=300, sigma=1.0, min_size=int(150 * sc * sc))
    return cv2.resize(lab.astype(np.int32), (frame.shape[1], frame.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------------
# 4. mean shift
# ----------------------------------------------------------------------

def seg_meanshift(frame, roi):
    """Cluster pixels by colour AND position, so nearby pixels can join the same group."""
    from sklearn.cluster import MeanShift, estimate_bandwidth
    h0, w0 = frame.shape[:2]
    sc = min(1.0, 160.0 / w0)                       # cap the width, this one is slow
    small = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # x and y are scaled by 0.3 so position matters, but colour still matters more.
    feat = np.dstack([lab, xx[..., None] * 0.3, yy[..., None] * 0.3]).reshape(-1, 5)
    bw = max(estimate_bandwidth(feat, quantile=0.12, n_samples=500), 5.0)
    ms = MeanShift(bandwidth=bw, bin_seeding=True, max_iter=30).fit(feat)
    lab_img = ms.labels_.reshape(h, w).astype(np.int32)
    return cv2.resize(lab_img, (w0, h0), interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------------
# 5. normalized cut
# ----------------------------------------------------------------------

def seg_ncut(frame, roi):
    """Cut the picture into superpixels, then cut the superpixel graph to minimise cut cost."""
    from skimage.segmentation import slic
    try:
        from skimage.graph import rag_mean_color, cut_normalized      # scikit-image >= 0.19
    except ImportError:
        from skimage.future.graph import rag_mean_color, cut_normalized
    sc = min(1.0, 320.0 / frame.shape[1])             # downsample, the graph step is heavy
    small = cv2.resize(frame, None, fx=sc, fy=sc)[..., ::-1]
    sp = slic(small, n_segments=250, compactness=10, start_label=1)
    g = rag_mean_color(small, sp, mode="similarity")
    lab = cut_normalized(sp, g, thresh=0.0002, num_cuts=8)
    return cv2.resize(lab.astype(np.int32), (frame.shape[1], frame.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


METHODS = [("Watershed", seg_watershed), ("Split & merge", seg_split_merge),
           ("Felzenszwalb", seg_felzenszwalb), ("Mean shift", seg_meanshift), ("N-Cut", seg_ncut)]


def main():
    import matplotlib
    matplotlib.use("Agg")            # no window, just save a picture
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description="Compare 5 segmentation methods on 5 frames.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--out", default=os.path.join("assets", "segmentation_grid.png"))
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    video = os.path.join(ROOT, a.video)
    roi = belt_roi.get_roi(video)
    print(f"[moduleA] impl={impl.get_mode()}  ROI={roi}")

    # Grab 5 frames spread over the middle 80 % of the video.
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(n * 0.1, n * 0.9, 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append((int(i), f))
    cap.release()

    fig, axes = plt.subplots(len(frames), 1 + len(METHODS),
                             figsize=(3.2 * (1 + len(METHODS)), 2.2 * len(frames)))
    axes = np.atleast_2d(axes)        # with only one row, matplotlib gives a flat array
    for c, name in enumerate(["Original"] + [m for m, _ in METHODS]):
        axes[0, c].set_title(name, fontsize=11)

    for r, (idx, frame) in enumerate(frames):
        axes[r, 0].imshow(frame[..., ::-1])          # BGR -> RGB for matplotlib
        axes[r, 0].set_ylabel(f"frame {idx}")
        for c, (name, fn) in enumerate(METHODS, 1):
            try:
                lab = fn(frame, roi)
                axes[r, c].imshow(colorize(lab, frame))
                print(f"frame {idx:4d} {name:<14} regions={len(np.unique(lab))}")
            except Exception as e:          # one broken method must not kill the whole grid
                axes[r, c].text(0.5, 0.5, f"{name}\nfailed:\n{type(e).__name__}",
                                ha="center", va="center")
                print(f"frame {idx:4d} {name:<14} FAILED: {e}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(ROOT, a.out)
    fig.savefig(out, dpi=90)
    print(f"[+] {out}")


if __name__ == "__main__":
    main()
