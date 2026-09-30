"""
Module D - Kalman filter written from scratch (NumPy only) + multi-box tracker.

The Kalman filter keeps one belief about one object:
    state   x = [x, y, vx, vy]   position + velocity, one step = one frame
    P       how unsure we are (a 4x4 covariance matrix)
    F H Q R the model matrices (constant velocity, we only SEE x and y)

Every frame does three steps:
    PREDICT   move the state forward with the last velocity   (x = F x,  P = F P F' + Q)
    MEASURE   the detector gives a noisy (x, y)               (called "update")
    UPDATE    blend the two, trusting the prediction more when P is large

When the detector sees nothing (occlusion) we only PREDICT, so the box keeps moving
even though we cannot see it.

    python moduleD/kalman_tracker.py --video assets/video/sd_conveyor_2.mp4
    python moduleD/kalman_tracker.py --impl alt

Outputs: assets/kalman_plot.png, assets/kalman_overlay.mp4

Alternative functions (see impl.py)
    base : np.linalg.inv for the 2x2 system, Joseph form for P, Hungarian matching
    alt  : closed-form 2x2 inverse, the short (I-KH)P form, greedy matching
"""
import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import belt_roi        # noqa: E402
import box_detect      # noqa: E402
import impl            # noqa: E402


# ----------------------------------------------------------------------
# alternatives: 2x2 inverse, P update, matching
# ----------------------------------------------------------------------

def inv2x2_numpy(S):
    """ORIGINAL: let NumPy invert the 2x2 matrix."""
    return np.linalg.inv(S)


def inv2x2_closed(S):
    """ALTERNATIVE: invert a 2x2 matrix by hand (faster and avoids a general solver).

        inv([[a, b], [c, d]]) = 1/(ad - bc) * [[d, -b], [-c, a]]
    """
    a, b = S[0, 0], S[0, 1]
    c, d = S[1, 0], S[1, 1]
    det = a * d - b * c
    det = det if abs(det) > 1e-12 else 1e-12
    return np.array([[d, -b], [-c, a]]) / det


def inv2x2(S):
    return inv2x2_closed(S) if impl.is_alt() else inv2x2_numpy(S)


def update_covariance_joseph(P, K, H, R):
    """ORIGINAL: the "Joseph form". Costs two extra matrix products but stays
    symmetric and positive, so P never goes negative or explodes."""
    I_KH = np.eye(4) - K @ H
    return I_KH @ P @ I_KH.T + K @ R @ K.T


def update_covariance_simple(P, K, H, R):
    """ALTERNATIVE: the short textbook form (I - K H) P. Two matrix products cheaper,
    but in floating point it can slowly lose its symmetry, and with a very confident
    filter (K close to identity) it is known to go unstable."""
    I_KH = np.eye(4) - K @ H
    return I_KH @ P


def update_covariance(P, K, H, R):
    """ORIGINAL = Joseph form (numerically safe), ALTERNATIVE = the short form (faster)."""
    return update_covariance_simple(P, K, H, R) if impl.is_alt() else update_covariance_joseph(P, K, H, R)


def match_hungarian(cost):
    """ORIGINAL matching: find the assignment that adds up cheapest overall.

    This can trade one good pair for two worse ones, which is exactly what we want
    when several boxes pass close to each other.
    """
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def match_greedy(cost):
    """ALTERNATIVE matching: always take the closest pair first, then the next closest.

    Simpler and faster, but it can be worse when two objects are close together:
    grabbing the single best pair can block a better overall set of pairs.
    """
    pairs, used_r, used_c = [], set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in order:
        r, c = int(r), int(c)
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c))
    return pairs


def match_pairs(cost):
    return match_greedy(cost) if impl.is_alt() else match_hungarian(cost)


# ----------------------------------------------------------------------
# the Kalman filter
# ----------------------------------------------------------------------

class KalmanFilter1:
    """Constant-velocity Kalman filter for ONE 2-D point."""

    def __init__(self, x0, y0, dt=1.0, q=0.5, r=4.0):
        self.x = np.array([x0, y0, 0.0, 0.0], float)
        self.P = np.diag([r, r, 100.0, 100.0])            # velocity unknown at start
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]], float)
        self.Q = q * (g @ g.T)                            # process noise (accel.)
        self.R = r * np.eye(2)                            # measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        z = np.asarray(z, float)
        y = z - self.H @ self.x                           # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv2x2(S)                # Kalman gain
        self.x = self.x + K @ y
        self.P = update_covariance(self.P, K, self.H, self.R)
        return self.x[:2].copy()


# ----------------------------------------------------------------------
# multi-box tracker
# ----------------------------------------------------------------------

class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.kf = KalmanFilter1(box["cx"], box["cy"])
        self.box = box
        self.hits = 1
        self.missed = 0
        self.sizes = []            # (rw, rh) of well-visible frames only
        self.types = []            # class votes
        self.vx = []               # filtered vx (px/frame)
        self.first_frame = None
        self.last_frame = None

    def add_measure(self, box, frame_idx):
        self.box = box
        self.hits += 1
        self.missed = 0
        self.last_frame = frame_idx
        if not box.get("touches_border", False):
            self.sizes.append((box["rw"], box["rh"]))
        if self.hits >= 5:
            self.vx.append(self.kf.x[2])

    def vote_type(self):
        v = [t for t in self.types if t >= 0]
        return Counter(v).most_common(1)[0][0] if v else -1     # MODE, not median

    def median_size(self):
        if not self.sizes:
            return None
        a = np.array(self.sizes)
        return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


class MultiTracker:
    def __init__(self, max_dist=100, max_missed=8, min_hits=3):
        self.tracks = {}
        self.finished = []
        self.next_id = 1
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.min_hits = min_hits

    def step(self, boxes, frame_idx):
        ids = list(self.tracks)
        preds = np.array([self.tracks[t].kf.predict() for t in ids]).reshape(-1, 2)
        matched_t, matched_b = set(), set()
        if ids and boxes:
            C = np.array([[np.hypot(b["cx"] - p[0], b["cy"] - p[1]) for b in boxes] for p in preds])
            for r, c in match_pairs(C):
                if C[r, c] < self.max_dist:
                    tr = self.tracks[ids[r]]
                    tr.kf.update([boxes[c]["cx"], boxes[c]["cy"]])
                    tr.add_measure(boxes[c], frame_idx)
                    matched_t.add(ids[r])
                    matched_b.add(c)
        for t in ids:
            if t not in matched_t:
                self.tracks[t].missed += 1
        for c, b in enumerate(boxes):                     # only UNMATCHED boxes start tracks
            if c not in matched_b:
                tr = Track(self.next_id, b)
                tr.first_frame = tr.last_frame = frame_idx
                self.tracks[self.next_id] = tr
                self.next_id += 1
        for t in [t for t, tr in self.tracks.items() if tr.missed > self.max_missed]:
            self.finished.append(self.tracks.pop(t))

    def confirmed(self):
        return [t for t in self.tracks.values() if t.hits >= self.min_hits and t.missed == 0]

    def all_tracks(self):
        return self.finished + list(self.tracks.values())


# ----------------------------------------------------------------------
# demo: follow ONE box, print the cycle, simulate occlusion, plot
# ----------------------------------------------------------------------

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description="Track one box with a Kalman filter.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--noise", type=float, default=0.0, help="extra Gaussian noise (px) on measurements")
    p.add_argument("--drop-len", type=int, default=5, help="occlusion length in frames (3-5)")
    p.add_argument("--drop-after", type=int, default=25, help="start dropout N frames after tracking starts")
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    video = os.path.join(ROOT, a.video)
    roi = belt_roi.get_roi(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    out_path = os.path.join(ROOT, "assets", "kalman_overlay.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    rng = np.random.default_rng(1)

    kf, trail = None, []
    rec = []                    # (frame, raw or None, filtered)
    gate = 0.15 * W
    started = None
    fi = -1
    print(f"[moduleD] impl={impl.get_mode()}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        boxes, vis = box_detect.segment_boxes(frame, roi)
        boxes = [b for b in boxes if not b["touches_border"]]

        raw = None
        if kf is None:
            if boxes:                                           # start on first clean box
                b = boxes[0]
                kf = KalmanFilter1(b["cx"], b["cy"])
                started = fi
                rec.append((fi, (b["cx"], b["cy"]), tuple(kf.x[:2])))
            writer.write(vis)
            continue

        pred = kf.predict()
        if boxes:                                               # nearest detection to prediction
            d = [np.hypot(b["cx"] - pred[0], b["cy"] - pred[1]) for b in boxes]
            j = int(np.argmin(d))
            if d[j] < gate:
                raw = np.array([boxes[j]["cx"], boxes[j]["cy"]]) + rng.normal(0, a.noise, 2)
        dropout = started + a.drop_after <= fi < started + a.drop_after + a.drop_len
        meas = None if dropout else raw
        if meas is not None:
            est = kf.update(meas)
        else:
            est = kf.x[:2].copy()                               # prediction only
        rec.append((fi, None if raw is None else tuple(raw), tuple(est), dropout))

        if started + a.drop_after - 4 <= fi < started + a.drop_after + a.drop_len + 4:   # cycle incl. dropout
            m = "DROPPED (occlusion)" if dropout else ("none" if raw is None else f"({raw[0]:7.1f},{raw[1]:6.1f})")
            print(f"frame {fi:4d} | predict ({pred[0]:7.1f},{pred[1]:6.1f}) | "
                  f"measure {m:>22} | update ({est[0]:7.1f},{est[1]:6.1f}) | v=({kf.x[2]:.2f},{kf.x[3]:.2f})")

        trail.append(tuple(int(v) for v in est))
        for t in trail[-40:]:
            cv2.circle(vis, t, 2, (0, 255, 0), -1)
        if raw is not None:
            cv2.circle(vis, (int(raw[0]), int(raw[1])), 5, (0, 0, 255), -1)
        cv2.circle(vis, (int(est[0]), int(est[1])), 6, (0, 255, 0), 2)
        if dropout:
            cv2.putText(vis, "OCCLUSION: prediction only", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(vis)
    cap.release()
    writer.release()

    if len(rec) < 5:
        raise SystemExit("[!] Not enough tracked frames for a plot.")
    fr = np.array([r[0] for r in rec])
    rx = np.array([r[1][0] if r[1] else np.nan for r in rec])
    ex = np.array([r[2][0] for r in rec])
    ry = np.array([r[1][1] if r[1] else np.nan for r in rec])
    ey = np.array([r[2][1] for r in rec])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (r_, e_, name) in zip(ax, [(rx, ex, "x"), (ry, ey, "y")]):
        axis.plot(fr, r_, "r.", label="raw detection")
        axis.plot(fr, e_, "g-", label="Kalman")
        axis.axvspan(started + a.drop_after, started + a.drop_after + a.drop_len,
                     color="gray", alpha=0.3, label="dropout")
        axis.set_xlabel("frame"); axis.set_ylabel(f"{name} (px)"); axis.legend()
    fig.suptitle("Raw measurements vs Kalman trajectory")
    plot_path = os.path.join(ROOT, "assets", "kalman_plot.png")
    fig.tight_layout(); fig.savefig(plot_path, dpi=120)
    print(f"[+] {plot_path}\n[+] {out_path}")


if __name__ == "__main__":
    main()
