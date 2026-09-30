"""
Module B part 1 of 3 - camera calibration with cv2.calibrateCamera.

What is calibration?
    The camera does not know how many millimetres one pixel covers. Calibration works
    that number out (plus the lens distortion) by looking at a checkerboard it knows the
    real size of, from many different angles.

How it is used here
    python moduleB/calibrate.py assets/reference_set --cols 10 --rows 7 --square-mm 24 \
           --out moduleB/calibration.json

What gets saved
    K        the camera matrix: fx, fy (focal length in px) and cx, cy (the optical centre)
    dist     lens distortion: k1, k2 (radial) and p1, p2 (tangential)
    rms      the average re-projection error in pixels. Under 1.0 px is good.
    rvec / tvec   where each photo stood in space, in mm

Very important detail
    K belongs to the resolution of the PHOTOS. If the video has another resolution,
    K must be scaled:  fx_new = fx_old * (video_width / photo_width).
    pipeline.py and measure_box.py do that scaling for you.

Alternative functions (see impl.py)
    base : corners are found with findChessboardCornersSB, falling back to the
           classic findChessboardCorners
    alt  : the classic finder runs FIRST with the adaptive-threshold flag, so very
           dark or very bright photos still work
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys                                             # noqa: E402
sys.path.insert(0, ROOT)
import impl                                            # noqa: E402


# ----------------------------------------------------------------------
# finding the checkerboard corners
# ----------------------------------------------------------------------

def find_corners_sb_first(gray, pattern):
    """ORIGINAL order: the fast "SB" finder, and only if it fails the classic one."""
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            # cornerSubPix slides a small window over the corner to find it more exactly.
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return ok, corners


def find_corners_classic_first(gray, pattern):
    """ALTERNATIVE order: classic finder with adaptive thresholding, SB finder as backup.

    ADAPTIVE_THRESH means each small window picks its own black/white threshold, so a
    photo with a bright corner or a shadow still finds its corners.
    """
    ok, corners = cv2.findChessboardCorners(
        gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        return ok, corners
    return cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_EXHAUSTIVE)


def find_corners(gray, pattern):
    """Find the corners of one checkerboard photo, chosen by the impl mode."""
    if impl.is_alt():
        return find_corners_classic_first(gray, pattern)
    return find_corners_sb_first(gray, pattern)


def list_photos(folder):
    """All picture files sitting DIRECTLY in this folder, sorted, board.png excluded."""
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        paths += glob.glob(os.path.join(folder, ext))
    # board.png is the PRINTED board, not a photo, so it must not be calibrated.
    return sorted(p for p in paths if os.path.basename(p) != "board.png")


def main():
    p = argparse.ArgumentParser(description="Calibrate the camera from checkerboard photos.")
    p.add_argument("images", nargs="?", default=os.path.join("assets", "checkerboard"))
    p.add_argument("--cols", type=int, default=9, help="number of SQUARES across (not corners)")
    p.add_argument("--rows", type=int, default=6, help="number of SQUARES down (not corners)")
    p.add_argument("--square-mm", type=float, default=24.0, help="real side of one square, in mm")
    p.add_argument("--out", default=os.path.join("moduleB", "calibration.json"))
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    # OpenCV wants the number of INNER corners, and a checkerboard of C x R squares
    # has (C-1) x (R-1) inner corners.
    pattern = (a.cols - 1, a.rows - 1)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    # The true 3-D position of every corner, in mm. z stays 0 (flat board).
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * a.square_mm

    paths = list_photos(a.images)
    print(f"{len(paths)} images in {a.images}   (impl={impl.get_mode()}, "
          f"inner corners {pattern[0]}x{pattern[1]})")
    if not paths:
        print("[!] No photos found. Put them DIRECTLY in this folder - subfolders are "
              "NOT scanned. Keep board.png (it is skipped on purpose, it is the "
              "printable board, not a photo).")

    obj_pts, img_pts, used, size = [], [], [], None
    unreadable = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            unreadable.append(path)
            print(f"[!] skip {os.path.basename(path)}: cannot read image "
                  f"(unsupported format? iPhone HEIC -> set Camera > Formats > Most Compatible)")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]                     # remember the photo size
        elif gray.shape[::-1] != size:
            print(f"[!] skip {os.path.basename(path)}: different size {gray.shape[::-1]} "
                  f"(expected {size})")
            continue                                    # a different size has a different K
        ok, corners = find_corners(gray, pattern)
        if not ok:
            print(f"[!] skip {os.path.basename(path)}: corners not found "
                  f"(blurry? board cut off? printed at wrong scale?)")
            continue
        obj_pts.append(objp)
        img_pts.append(corners.reshape(-1, 1, 2).astype(np.float32))
        used.append(path)
        print(f"[+] {os.path.basename(path)}")

    if len(used) < 3:
        if unreadable:
            print(f"\n{len(unreadable)} photo(s) could not be read at all - see the "
                  f"'cannot read image' lines above.")
        raise SystemExit("[!] Need at least 3 good photos (10-15 recommended).")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f"\nRMS re-projection error: {rms:.4f} px   (aim < 1.0)")
    print("K =\n", np.round(K, 3))
    print("dist (k1,k2,p1,p2,k3) =", np.round(dist.ravel(), 5).tolist())

    # Re-project every corner and measure how far it lands from where we found it.
    per_img = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_img.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2)))))

    data = {
        "pattern": {"squares_cols": a.cols, "squares_rows": a.rows}, "square_mm": a.square_mm,
        "image_size": {"width": size[0], "height": size[1]},
        "rms": float(rms), "K": K.tolist(), "dist": dist.ravel().tolist(), "n_images": len(used),
        "per_image_error_px": dict(zip([os.path.basename(u) for u in used], per_img)),
        "extrinsics": {os.path.basename(u): {"rvec": r.ravel().tolist(), "tvec": t.ravel().tolist()}
                       for u, r, t in zip(used, rvecs, tvecs)},
    }
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[+] saved {out}")

    # Before / after picture so the effect of undistortion is visible.
    img = cv2.imread(used[0])
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    und = cv2.undistort(img, K, dist, None, newK)
    combo = np.hstack([img, und])
    cv2.putText(combo, "BEFORE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(combo, "AFTER", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    demo = os.path.join(ROOT, "assets", "undistort_demo.png")
    cv2.imwrite(demo, combo)
    print(f"[+] {demo}")


if __name__ == "__main__":
    main()
