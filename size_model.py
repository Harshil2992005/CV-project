"""
size_model.py - pixels -> centimetres, per box (fixes B6, B7, B8).

Old code: ONE number px_per_cm for the whole frame. With a strong perspective a far box and
a near box have different scales, and a tall box (top face close to the camera) looks bigger
than a flat one. This file has three levels, best first:

  Camera3D     needs the camera matrix K (moduleB/calibrate.py of YOUR camera) and ONE reference
               box of known W x D x H. The reference's top face gives the camera pose (PnP).
               Every other box: the bottom of its front edge sits on the belt (z = 0), so its
               footprint is known exactly; that fixes its HEIGHT; then the top face is measured
               on the plane at that height. Per-box depth, tall and flat boxes both correct.
  PlaneScale   no K needed. Reference top face -> homography to a W x D rectangle. Exact for
               boxes of the SAME height as the reference, perspective across the frame included.
               Other heights: pass --cam-height-cm and --box-height-cm to correct (optional).
  (legacy)     one px_per_cm (--px-per-cm), only a rough approximation.

SizeEstimator keeps the measurements of one track and returns a robust size with a spread, so
one wrong frame (merged boxes, cut-off box, motion blur) cannot decide the answer.
"""
import json

import cv2
import numpy as np


def _edge_dims(P):
    """(long, short) from AVERAGE of opposite edges of a 4-point polygon (any units)."""
    a = (np.linalg.norm(P[1] - P[0]) + np.linalg.norm(P[2] - P[3])) / 2.0
    b = (np.linalg.norm(P[3] - P[0]) + np.linalg.norm(P[2] - P[1])) / 2.0
    return (max(a, b), min(a, b))


def _order(q):
    q = np.asarray(q, np.float32).reshape(4, 2)
    s, d = q.sum(1), np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


# ---------------------------------------------------------------------------
class PlaneScale:
    """Image -> centimetres on ONE plane, from the top face of a reference box."""

    def __init__(self, Hm, ref_height_cm=0.0, cam_height_cm=0.0):
        self.H = Hm
        self.ref_h = ref_height_cm
        self.cam_h = cam_height_cm

    @classmethod
    def from_reference(cls, top_quad_px, W_cm, D_cm, ref_height_cm=0.0, cam_height_cm=0.0):
        q = _order(top_quad_px)
        pl, ps = _edge_dims(q)
        L, S = (max(W_cm, D_cm), min(W_cm, D_cm))
        # long side of the rectangle <-> long side in the picture
        a = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
        b = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
        w, d = (L, S) if a >= b else (S, L)
        dst = np.float32([[0, d], [w, d], [w, 0], [0, 0]])      # far edge = +y, so x right, y far, z up
        return cls(cv2.getPerspectiveTransform(q, dst), ref_height_cm, cam_height_cm)

    def to_cm(self, pts):
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def measure(self, top_quad_px, box_height_cm=None):
        L, S = _edge_dims(self.to_cm(_order(top_quad_px)))
        if box_height_cm is not None and self.cam_h > 0 and box_height_cm < self.cam_h:
            k = (self.cam_h - box_height_cm) / max(self.cam_h - self.ref_h, 1e-6)
            L, S = L * k, S * k          # a taller top face looks bigger -> shrink
        return float(L), float(S), (None if box_height_cm is None else float(box_height_cm))


# ---------------------------------------------------------------------------
class Camera3D:
    """Pinhole camera with a pose from ONE reference box -> per-box height and top-face size."""

    def __init__(self, K, rvec, tvec):
        self.K = np.asarray(K, np.float64)
        self.rvec, self.tvec = rvec, tvec
        self.R, _ = cv2.Rodrigues(rvec)
        self.C = (-self.R.T @ tvec).ravel()          # camera centre in the reference frame (cm)

    @classmethod
    def from_reference(cls, K, top_quad_px, W_cm, D_cm, H_cm):
        q = _order(top_quad_px)
        pl, ps = _edge_dims(q)
        L, S = (max(W_cm, D_cm), min(W_cm, D_cm))
        a = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
        b = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
        w, d = (L, S) if a >= b else (S, L)
        obj = np.float64([[0, d, 0], [w, d, 0], [w, 0, 0], [0, 0, 0]])   # IPPE wants a z=0 plane; height is added below
        Kf = np.asarray(K, np.float64)
        n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, q.astype(np.float64), Kf, None, flags=cv2.SOLVEPNP_IPPE)
        best = None
        for rv, tv, e in zip(rvs, tvs, np.ravel(errs)):
            R, _ = cv2.Rodrigues(rv)
            tv = tv - R @ np.array([[0.0], [0.0], [H_cm]])               # the top face is H above the belt
            C = (-R.T @ tv).ravel()
            if C[2] > H_cm + 1.0 and (best is None or e < best[2]):      # camera must be ABOVE the box
                best = (rv, tv, e)
        if best is None:
            return None
        return cls(K, best[0], best[1])

    @classmethod
    def from_reference_frames(cls, K, quads, W_cm, D_cm, H_cm):
        """Same, but from MANY frames of the moving reference box (much steadier pose).

        The belt is straight, so between frames the reference only slides along x: the camera's
        rotation, its height and its distance from the belt line do not change. Take the median
        of those, and set the (meaningless) position along the belt to 0.
        """
        cams = [cls.from_reference(K, q, W_cm, D_cm, H_cm) for q in quads]
        cams = [c for c in cams if c is not None]
        if not cams:
            return None
        if len(cams) < 3:
            return cams[0]
        Rm = np.median(np.stack([c.R for c in cams]), axis=0)
        u, _, vt = np.linalg.svd(Rm)                       # nearest true rotation matrix
        R = u @ vt
        if np.linalg.det(R) < 0:
            R = u @ np.diag([1, 1, -1]) @ vt
        Cy = float(np.median([c.C[1] for c in cams]))
        Cz = float(np.median([c.C[2] for c in cams]))
        C = np.array([0.0, Cy, Cz])
        rvec, _ = cv2.Rodrigues(R)
        return cls(K, rvec, (-R @ C).reshape(3, 1))

    def ray_plane(self, px, z):
        """3D point (x, y, z) where the viewing ray of pixel px meets the plane at height z."""
        p = np.asarray(px, np.float64).reshape(-1, 2)
        d = (np.linalg.inv(self.K) @ np.c_[p, np.ones(len(p))].T)       # camera coords
        dw = self.R.T @ d                                                # world dirs
        t = (z - self.C[2]) / dw[2]
        return (self.C[:, None] + dw * t).T

    def project(self, P):
        p, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 1, 3), self.rvec, self.tvec, self.K, None)
        return p.reshape(-1, 2)

    def height_from_front_edge(self, top_quad_px, sil_quad_px):
        """Height of a box: its front-bottom corners lie on the belt (z=0).

        The two LOW corners of the top face (largest y in the picture) sit above the two low
        corners of the silhouette. Back-project the silhouette corners to z=0 -> (X, Y); the
        height h is the value for which (X, Y, h) projects onto the top-face corner.
        """
        tq = _order(top_quad_px)
        sq = _order(sil_quad_px)
        hs = []
        for ti, si in ((3, 3), (2, 2)):                    # bl, br
            P0 = self.ray_plane(sq[si], 0.0)[0]
            best, bh = 1e18, None
            for h in np.linspace(2.0, 120.0, 238):
                e = np.linalg.norm(self.project([[P0[0], P0[1], h]])[0] - tq[ti])
                if e < best:
                    best, bh = e, h
            if best < 8.0:
                hs.append(bh)
        return float(np.median(hs)) if hs else None

    def measure(self, top_quad_px, sil_quad_px=None, height_cm=None):
        h = height_cm
        if h is None and sil_quad_px is not None:
            h = self.height_from_front_edge(top_quad_px, sil_quad_px)
        if h is None:
            return None
        P = self.ray_plane(_order(top_quad_px), h)[:, :2]
        L, S = _edge_dims(P)
        return float(L), float(S), float(h)


# ---------------------------------------------------------------------------
class SizeEstimator:
    """Robust size of ONE tracked box from many per-frame measurements (cm or px)."""

    def __init__(self):
        self.m = []            # (L, S, h)

    def add(self, L, S, h=None):
        self.m.append((L, S, np.nan if h is None else h))

    def result(self, min_n=5):
        if len(self.m) < min_n:
            return None
        a = np.array(self.m, float)
        keep = np.ones(len(a), bool)
        for _ in range(2):                       # two rounds of MAD outlier rejection
            med = np.nanmedian(a[keep, :2], axis=0)
            mad = np.nanmedian(np.abs(a[keep, :2] - med), axis=0) * 1.4826 + 1e-6
            keep = (np.abs(a[:, :2] - med) <= 3.0 * np.maximum(mad, 0.02 * med)).all(axis=1)
            if keep.sum() < min_n:
                keep[:] = True
                break
        L, S = np.median(a[keep, 0]), np.median(a[keep, 1])
        h = float(np.nanmedian(a[keep, 2])) if np.isfinite(a[keep, 2]).any() else None
        return {"L": float(L), "S": float(S), "H": h, "n": int(keep.sum()), "n_total": len(a),
                "spread_L": float(np.std(a[keep, 0])), "spread_S": float(np.std(a[keep, 1]))}


def load_K(calib_json, video_w=None):
    with open(calib_json) as f:
        c = json.load(f)
    K = np.array(c["K"], np.float64)
    if video_w:
        k = video_w / c["image_size"]["width"]
        K[0] *= k
        K[1] *= k
    return K
