"""
make_synthetic.py - render a FIXED-camera conveyor video with KNOWN box sizes (ground truth).

Why: the real clips have no ground truth, so nobody can say "the size is 30.2 cm, true 30 cm".
Here the camera is fixed, the perspective is strong (camera looks down at an angle), boxes
are real 3D cuboids with a top face, a front face and a side face, some of them touch. The
true sizes and the true top-face corners of every box in every frame are written to a json.

    python tools/make_synthetic.py --out assets/video/synthetic_fixed.mp4
Outputs: <out>.mp4 and <out>.gt.json
"""
import argparse
import json
import os

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
K = np.array([[1000, 0, W / 2], [0, 1000, H / 2], [0, 0, 1]], np.float64)

# (id, L along belt, D across belt, height, y centre, x start (cm), colour BGR)
BOXES = [
    (1, 30, 20, 15, -4, -150, (112, 160, 196)),
    (2, 25, 25, 25, 6, -150 - 80, (100, 150, 190)),
    (3, 40, 18, 12, -6, -300, (120, 165, 200)),
    (4, 20, 15, 10, 8, -300 - 30, (105, 155, 192)),    # touches box 3 (centre distance = half the sum of lengths)
    (5, 35, 25, 20, 0, -450, (110, 158, 195)),
]
SPEED_CM_F = 0.9           # belt speed in cm per frame


def camera():
    C = np.array([0.0, -105.0, 85.0])
    T = np.array([0.0, 0.0, 8.0])
    z = T - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])       # right
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                 # down
    R = np.stack([x, y, z])
    t = -R @ C
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t.reshape(3, 1), C


def proj(P, rvec, t):
    p, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 1, 3), rvec, t, K, None)
    return p.reshape(-1, 2)


def cuboid(cx, cy, L, D, Ht):
    x0, x1, y0, y1 = cx - L / 2, cx + L / 2, cy - D / 2, cy + D / 2
    return {
        "top":   [(x0, y0, Ht), (x1, y0, Ht), (x1, y1, Ht), (x0, y1, Ht)],
        "front": [(x0, y0, 0), (x1, y0, 0), (x1, y0, Ht), (x0, y0, Ht)],     # faces camera (y<0)
        "left":  [(x0, y0, 0), (x0, y1, 0), (x0, y1, Ht), (x0, y0, Ht)],
        "right": [(x1, y0, 0), (x1, y1, 0), (x1, y1, Ht), (x1, y0, Ht)],
    }


def render(n_frames, seed=0, hard=False):
    rng = np.random.RandomState(seed)
    rvec, t, C = camera()
    gt = {"fps": FPS, "K": K.tolist(), "cam_height_cm": float(C[2]), "boxes": {}, "frames": []}
    for b in BOXES:
        gt["boxes"][str(b[0])] = {"L": b[1], "D": b[2], "H": b[3]}
    belt = proj([(-400, -30, 0), (400, -30, 0), (400, 30, 0), (-400, 30, 0)], rvec, t).astype(np.int32)
    for f in range(n_frames):
        img = np.zeros((H, W, 3), np.uint8)
        yy = np.linspace(0, 1, H)[:, None, None]
        img[:] = (np.array([70, 74, 78]) * (1 - yy) + np.array([45, 48, 52]) * yy).astype(np.uint8)
        cv2.fillPoly(img, [belt], (92, 96, 96))
        # belt texture: slats moving with the belt
        for s in range(-40, 41):
            x = s * 10 + (f * SPEED_CM_F) % 10
            a, bq = proj([(x, -30, 0), (x, 30, 0)], rvec, t)
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in bq), (82, 86, 86), 2)
        for e in (-30, 30):
            a, bq = proj([(-400, e, 0), (400, e, 0)], rvec, t)
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in bq), (60, 62, 64), 6)
        items = []
        for (bid, L, D, Ht, yc, x0, col) in BOXES:
            cx = x0 + f * SPEED_CM_F
            if cx < -110 or cx > 110:
                continue
            items.append((bid, cx, L, D, Ht, yc, col))
        fr = {"i": f, "boxes": []}
        for (bid, cx, L, D, Ht, yc, col) in sorted(items, key=lambda r: -r[5]):
            faces = cuboid(cx, yc, L, D, Ht)
            sh = np.array([(cx - L / 2 + 6, yc - D / 2 + 3, 0), (cx + L / 2 + 6, yc - D / 2 + 3, 0),
                           (cx + L / 2 + 6, yc + D / 2 + 3, 0), (cx - L / 2 + 6, yc + D / 2 + 3, 0)])
            ov = img.copy()
            cv2.fillPoly(ov, [proj(sh, rvec, t).astype(np.int32)], (50, 52, 52))
            img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
            shade = {"top": 1.0, "front": 0.78, "left": 0.62, "right": 0.62}
            for name in ("left", "right", "front", "top"):
                if name == "left" and cx - L / 2 < 0:
                    continue                      # camera is at x=0: only the face turned to it
                if name == "right" and cx + L / 2 > 0:
                    continue
                poly = proj(faces[name], rvec, t)
                c = tuple(int(min(255, v * shade[name])) for v in col)
                cv2.fillPoly(img, [poly.astype(np.int32)], c)
                cv2.polylines(img, [poly.astype(np.int32)], True, tuple(int(v * 0.7) for v in c), 1)
            tape = proj([(cx - L / 2, yc - 1.5, Ht), (cx + L / 2, yc - 1.5, Ht),
                         (cx + L / 2, yc + 1.5, Ht), (cx - L / 2, yc + 1.5, Ht)], rvec, t)
            cv2.fillPoly(img, [tape.astype(np.int32)], tuple(int(v * 0.8) for v in col))
            if hard:        # a white label on the top face and a dark strip on the front face
                lab = proj([(cx - L / 4, yc - D / 5, Ht), (cx + L / 8, yc - D / 5, Ht),
                            (cx + L / 8, yc + D / 5, Ht), (cx - L / 4, yc + D / 5, Ht)], rvec, t)
                cv2.fillPoly(img, [lab.astype(np.int32)], (225, 225, 230))
                st = proj([(cx - L / 2, yc - D / 2, Ht * 0.35), (cx + L / 2, yc - D / 2, Ht * 0.35),
                           (cx + L / 2, yc - D / 2, Ht * 0.2), (cx - L / 2, yc - D / 2, Ht * 0.2)], rvec, t)
                cv2.fillPoly(img, [st.astype(np.int32)], (60, 80, 100))
            fr["boxes"].append({"id": bid, "top_quad": proj(faces["top"], rvec, t).tolist(),
                                "cx_cm": float(cx)})
        img = cv2.GaussianBlur(img, (3, 3), 0)
        if hard:        # motion blur along the belt, exposure flicker, heavier sensor noise
            k = np.zeros((1, 9), np.float32); k[:] = 1.0 / 9
            img = cv2.filter2D(img, -1, k)
            img = np.clip(img.astype(np.float32) * (1 + 0.06 * np.sin(f / 7.0)), 0, 255)
            img = np.clip(img + rng.normal(0, 6.0, img.shape), 0, 255).astype(np.uint8)
        else:
            img = np.clip(img.astype(np.float32) + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
        gt["frames"].append(fr)
        yield img, gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("assets", "video", "synthetic_fixed.mp4"))
    p.add_argument("--frames", type=int, default=720)
    p.add_argument("--hard", action="store_true", help="labels, dark strips, motion blur, flicker, more noise")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = None
    for img, gt in render(a.frames, hard=a.hard):
        wr.write(img)
    wr.release()
    with open(os.path.splitext(a.out)[0] + ".gt.json", "w") as f:
        json.dump(gt, f)
    with open(os.path.splitext(a.out)[0] + ".calib.json", "w") as f:       # same format as moduleB/calibration.json
        json.dump({"K": K.tolist(), "image_size": {"width": W, "height": H}, "rms": 0.0}, f)
    print("[+] wrote", a.out, "frames:", len(gt["frames"]))


if __name__ == "__main__":
    main()
