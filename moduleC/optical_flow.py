"""
Module C - Optical flow. How fast is the belt moving?

Optical flow answers: if a small patch of pixels looks the same as it did a moment
ago, where did it go? That gives us how fast it moved, in pixels per frame.

We have two types of LK flow (Lucas-Kanade):
  (a) DENSE flow, hand-written (solve the 2x2 normal equations on every pixel
      inside a window, using a coarse-to-fine pyramid)
  (b) SPARSE flow (track corners): OpenCV goodFeaturesToTrack + calcOpticalFlowPyrLK

The aperture problem
    The brightness equation gives 1 equation with 2 unknowns. If a patch is FLAT (no
    corners inside it), we cannot tell if it slid left or diagonally - the 2x2 matrix
    becomes almost singular. We fix that by only trusting pixels where that matrix
    has a large enough smallest eigenvalue (the "min eigenvalue test"). Sparse flow
    avoids this entirely by only tracking corners, which are never flat.

Then we fit an AFFINE flow model to the box region using RANSAC
    u = a1 + a2*x + a3*y    v = a4 + a5*x + a6*y
    That averages over many points, throws away outliers (blurry corners) and gives one
    clean (dx, dy) for the whole region.

    python moduleC/optical_flow.py --video assets/video/sd_conveyor_2.mp4
    python moduleC/optical_flow.py --px-per-cm 20.5
    python moduleC/optical_flow.py --impl alt

Outputs: assets/flow_arrows.png, assets/flow_sparse.png, assets/flow_affine.png
Speed:   printed at the end (px/s or cm/s)

Alternative functions (see impl.py)
    base : dense LK written by hand with cv2.Sobel + cv2.boxFilter
    alt  : dense LK but gradients computed with scikit-image sobel + uniform_filter
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import belt_roi        # noqa: E402
import box_detect      # noqa: E402
import impl            # noqa: E402


# ----------------------------------------------------------------------
# (a) gradients
# ----------------------------------------------------------------------

def gradients_cv2(I1):
    """ORIGINAL: OpenCV Sobel to get Ix, Iy. It is fast and standard."""
    Ix = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return Ix, Iy


def gradients_skimage(I1):
    """ALTERNATIVE: scikit-image sobel to get Ix, Iy. Same math, different library."""
    from skimage.filters import sobel
    Ix = sobel(I1, axis=1).astype(np.float32)
    Iy = sobel(I1, axis=0).astype(np.float32)
    return Ix, Iy


def gradients(I1):
    return gradients_skimage(I1) if impl.is_alt() else gradients_cv2(I1)


# ----------------------------------------------------------------------
# (a) dense LK from scratch
# ----------------------------------------------------------------------

def _lk_residual(I1, I2w, win, min_eig):
    """Solve [Sxx Sxy; Sxy Syy][u v]^T = -[Sxt Syt]^T in a window around every pixel."""
    Ix, Iy = gradients(I1)
    It = I2w - I1
    box = lambda a: cv2.boxFilter(a, cv2.CV_32F, (win, win), normalize=False)
    Sxx, Sxy, Syy = box(Ix * Ix), box(Ix * Iy), box(Iy * Iy)
    Sxt, Syt = box(Ix * It), box(Iy * It)
    det = Sxx * Syy - Sxy * Sxy
    tr = Sxx + Syy
    lam_min = tr / 2 - np.sqrt(np.maximum((Sxx - Syy) ** 2 / 4 + Sxy * Sxy, 0))
    valid = lam_min > min_eig * win * win
    det = np.where(valid, det, 1.0)
    u = (-Syy * Sxt + Sxy * Syt) / det
    v = (Sxy * Sxt - Sxx * Syt) / det
    u = np.where(valid, u, 0.0)
    v = np.where(valid, v, 0.0)
    return u, v, valid


def lucas_kanade_dense(prev_gray, curr_gray, win=15, levels=4, iters=3, min_eig=1e-3):
    """Coarse-to-fine dense LK. Returns (flow (H,W,2), validity mask)."""
    I1 = prev_gray.astype(np.float32) / 255.0
    I2 = curr_gray.astype(np.float32) / 255.0
    p1, p2 = [I1], [I2]
    for _ in range(levels - 1):
        p1.append(cv2.pyrDown(p1[-1]))
        p2.append(cv2.pyrDown(p2[-1]))
    flow = np.zeros(p1[-1].shape + (2,), np.float32)
    for lvl in range(levels - 1, -1, -1):
        A, B = p1[lvl], p2[lvl]
        if flow.shape[:2] != A.shape:
            flow = cv2.resize(flow, (A.shape[1], A.shape[0])) * 2.0
        gy, gx = np.mgrid[0:A.shape[0], 0:A.shape[1]].astype(np.float32)
        for _ in range(iters):
            Bw = cv2.remap(B, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            du, dv, valid = _lk_residual(A, Bw, win, min_eig)
            flow[..., 0] += du
            flow[..., 1] += dv
    return flow, valid


def draw_flow_arrows(img, flow, valid, step=16, scale=3.0, color=(0, 255, 0), roi=None):
    vis = img.copy()
    H, W = flow.shape[:2]
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            if not valid[y, x]:
                continue
            if roi and not (roi["x"] <= x <= roi["x"] + roi["w"] and roi["y"] <= y <= roi["y"] + roi["h"]):
                continue
            dx, dy = flow[y, x]
            if abs(dx) + abs(dy) < 0.3:
                continue
            cv2.arrowedLine(vis, (x, y), (int(x + dx * scale), int(y + dy * scale)), color, 1, tipLength=0.3)
    return vis


# ----------------------------------------------------------------------
# (b) sparse flow
# ----------------------------------------------------------------------

def sparse_flow(prev_gray, curr_gray, mask=None, max_corners=300):
    pts = cv2.goodFeaturesToTrack(prev_gray, max_corners, 0.01, 7, mask=mask)
    if pts is None:
        return np.empty((0, 2)), np.empty((0, 2))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None,
                                          winSize=(21, 21), maxLevel=3)
    st = st.ravel() == 1
    p0 = pts.reshape(-1, 2)[st]
    p1 = nxt.reshape(-1, 2)[st]
    return p0, p1 - p0


# ----------------------------------------------------------------------
# (c) affine RANSAC
# ----------------------------------------------------------------------

def _design(pts):
    n = len(pts)
    A = np.zeros((2 * n, 6))
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = 1, pts[:, 0], pts[:, 1]
    A[1::2, 3], A[1::2, 4], A[1::2, 5] = 1, pts[:, 0], pts[:, 1]
    return A


def fit_affine(pts, flow):
    A = _design(pts)
    b = flow.reshape(-1)
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    return params


def fit_affine_ransac_numpy(pts, flow, iters=300, thr=0.6, seed=0, origin=None):
    """ORIGINAL: simple RANSAC implemented with NumPy loops.

    `origin` is accepted so both versions have the same signature, but the NumPy
    version always fits on the raw pixel coordinates, exactly like the original code.
    """
    n = len(pts)
    if n < 3:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best_inl = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            prm = fit_affine(pts[idx], flow[idx])
        except np.linalg.LinAlgError:
            continue
        res = (_design(pts) @ prm - flow.reshape(-1)).reshape(-1, 2)
        inl = np.hypot(res[:, 0], res[:, 1]) < thr
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < 3:
        return None, np.zeros(n, bool)
    return fit_affine(pts[best_inl], flow[best_inl]), best_inl


def fit_affine_ransac_cv2(pts, flow, iters=300, thr=0.6, seed=0, origin=None):
    """ALTERNATIVE: OpenCV estimateAffine2D with RANSAC (needs >= 3 points).

    One extra step is needed here, and it is the whole point of this alternative.
    estimateAffine2D builds the model
        u = a1 + a2*x + a3*y
    straight out of the raw pixel coordinates. When the motion is nearly a pure
    translation, a2 and a3 are decided by noise alone. Reading the answer at the
    centre of the belt then means evaluating 1.0004 * 384, which turns a 0.03 px
    mistake into a 150 px one. That really happens, it is not theoretical.

    The fix is to fit on coordinates that are CENTRED ON THE POINT WHERE WE WANT TO
    READ THE ANSWER. After that shift
        u = a1 + a2*(x - x0) + a3*(y - y0)
    a1 is the flow at (x0, y0) directly, and a2 / a3 only say how much the speed
    changes across the frame, so they cannot spoil a1.

    `origin` is the point the answer will be read at. If it is not given, the centre
    of the point cloud is used. The returned parameters are converted back, so they
    mean exactly the same thing as the NumPy version above.
    """
    if len(pts) < 3:
        return None, np.zeros(len(pts), bool)
    c = np.asarray(origin, float) if origin is not None else pts.mean(axis=0)
    p = pts - c
    pts0 = p.astype(np.float32).reshape(-1, 1, 2)
    pts1 = (p + flow).astype(np.float32).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffine2D(pts0, pts1, method=cv2.RANSAC,
                                  ransacReprojThreshold=thr, maxIters=iters, confidence=0.99)
    if M is None:
        return None, np.zeros(len(pts), bool)
    inl = inl.ravel() == 1
    if inl.sum() < 3:
        return None, np.zeros(len(pts), bool)
    # Undo the centring: u = M02 + M00*(x - c0) + M01*(y - c1)
    params = np.array([M[0, 2] - M[0, 0] * c[0] - M[0, 1] * c[1], M[0, 0], M[0, 1],
                       M[1, 2] - M[1, 0] * c[0] - M[1, 1] * c[1], M[1, 0], M[1, 1]])
    return params, inl


def fit_affine_ransac(pts, flow, iters=300, thr=0.6, seed=0, origin=None):
    if impl.is_alt():
        return fit_affine_ransac_cv2(pts, flow, iters, thr, seed, origin)
    return fit_affine_ransac_numpy(pts, flow, iters, thr, seed, origin)


def affine_flow_at(params, x, y):
    return params[0] + params[1] * x + params[2] * y, params[3] + params[4] * x + params[5] * y


def estimate_belt_motion(prev_gray, curr_gray, roi=None):
    mask = None
    cx = cy = None
    if roi is not None:
        mask = np.zeros(prev_gray.shape, np.uint8)
        mask[roi["y"]:roi["y"] + roi["h"], roi["x"]:roi["x"] + roi["w"]] = 255
        cx, cy = roi["x"] + roi["w"] / 2, roi["y"] + roi["h"] / 2
    p0, fl = sparse_flow(prev_gray, curr_gray, mask)
    if len(p0) < 6:
        return None
    if cx is None:
        cx, cy = prev_gray.shape[1] / 2, prev_gray.shape[0] / 2
    prm, inl = fit_affine_ransac(p0, fl, origin=(cx, cy))
    if prm is None:
        return None
    return affine_flow_at(prm, cx, cy)


def px_per_cm_from_args(args, video_w):
    if args.px_per_cm:
        return args.px_per_cm
    if args.calib and args.cam_height_cm:
        with open(args.calib) as f:
            c = json.load(f)
        fx = c["K"][0][0] * video_w / c["image_size"]["width"]
        return fx / args.cam_height_cm
    return None


def main():
    p = argparse.ArgumentParser(description="Measure belt speed with optical flow.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--frame", type=int, default=-1, help="first frame of the pair (-1 = middle)")
    p.add_argument("--gap", type=int, default=1, help="frames between the pair")
    p.add_argument("--px-per-cm", type=float, default=0.0)
    p.add_argument("--calib", default="", help="calibration.json (needs --cam-height-cm)")
    p.add_argument("--cam-height-cm", type=float, default=0.0)
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    video = os.path.join(ROOT, a.video)
    roi = belt_roi.get_roi(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = a.frame if a.frame >= 0 else n // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok1, fr1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0 + a.gap)
    ok2, fr2 = cap.read()
    cap.release()
    if not (ok1 and ok2):
        raise SystemExit("[!] Could not read the frame pair.")
    g1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
    out = os.path.join(ROOT, "assets")
    print(f"[moduleC] impl={impl.get_mode()}")

    flow, valid = lucas_kanade_dense(g1, g2)
    cv2.imwrite(os.path.join(out, "flow_arrows.png"), draw_flow_arrows(fr1, flow, valid, roi=roi))

    mask = np.zeros(g1.shape, np.uint8)
    mask[roi["y"]:roi["y"] + roi["h"], roi["x"]:roi["x"] + roi["w"]] = 255
    p0, fl = sparse_flow(g1, g2, mask)
    vis = fr1.copy()
    for (x, y), (dx, dy) in zip(p0, fl):
        cv2.arrowedLine(vis, (int(x), int(y)), (int(x + dx * 3), int(y + dy * 3)), (255, 128, 0), 1, tipLength=0.3)
    cv2.imwrite(os.path.join(out, "flow_sparse.png"), vis)

    boxes, _ = box_detect.segment_boxes(fr1, roi)
    boxes = [b for b in boxes if not b["touches_border"]]
    if boxes:
        b = max(boxes, key=lambda b: b["w"] * b["h"])
        reg = {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
    else:
        reg = roi
    m2 = np.zeros(g1.shape, np.uint8)
    m2[reg["y"]:reg["y"] + reg["h"], reg["x"]:reg["x"] + reg["w"]] = 255
    q0, qf = sparse_flow(g1, g2, m2, max_corners=200)
    if len(q0) < 6:
        ys, xs = np.where(valid & (m2 > 0))
        if len(xs):
            sel = np.arange(0, len(xs), max(1, len(xs) // 300))
            q0 = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
            qf = flow[ys[sel], xs[sel]]
    reg_cx, reg_cy = reg["x"] + reg["w"] / 2, reg["y"] + reg["h"] / 2
    prm, inl = fit_affine_ransac(q0, qf, origin=(reg_cx, reg_cy))
    vis = fr1.copy()
    cv2.rectangle(vis, (reg["x"], reg["y"]), (reg["x"] + reg["w"], reg["y"] + reg["h"]), (0, 255, 255), 2)
    for (x, y), (dx, dy), ok in zip(q0, qf, inl):
        col = (0, 255, 0) if ok else (0, 0, 255)
        cv2.arrowedLine(vis, (int(x), int(y)), (int(x + dx * 3), int(y + dy * 3)), col, 2, tipLength=0.3)
    cv2.imwrite(os.path.join(out, "flow_affine.png"), vis)

    if prm is None:
        raise SystemExit("[!] Affine fit failed (too few flow vectors).")
    u, v = affine_flow_at(prm, reg_cx, reg_cy)
    px_frame = float(np.hypot(u, v)) / a.gap
    px_s = px_frame * fps
    print(f"Affine params (a1..a6): {np.round(prm, 4).tolist()}")
    print(f"Inliers: {int(inl.sum())}/{len(inl)}   flow at region centre: ({u:.2f}, {v:.2f}) px/{a.gap}frame")
    pcm = px_per_cm_from_args(a, fr1.shape[1])
    if pcm:
        print(f"BELT SPEED: {px_s / pcm:.2f} cm/s   ({px_s:.1f} px/s, {pcm:.2f} px/cm)")
    else:
        print(f"BELT SPEED: {px_s:.1f} px/s   (give --px-per-cm or --calib + --cam-height-cm for cm/s)")
    print("[+] saved flow_arrows.png, flow_sparse.png, flow_affine.png in assets/")


if __name__ == "__main__":
    main()
