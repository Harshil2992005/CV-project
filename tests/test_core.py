"""Self-checks for the new geometry code:  python -m unittest tests.test_core -v   (from conveyor_v2/)"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shape_fit      # noqa: E402
import size_model     # noqa: E402


def rect(h, w, boxes):
    m = np.zeros((h, w), np.uint8)
    for (x, y, bw, bh) in boxes:
        m[y:y + bh, x:x + bw] = 255
    return m


class TestShape(unittest.TestCase):
    def test_single_box_not_split(self):
        m = rect(200, 300, [(40, 40, 160, 100)])
        self.assertEqual(len(shape_fit.split_blob(m, None, 100)), 1)

    def test_perspective_quad_is_one_box(self):
        m = np.zeros((200, 300), np.uint8)
        cv2.fillPoly(m, [np.int32([[60, 50], [220, 50], [250, 150], [30, 150]])], 255)
        parts = shape_fit.split_blob(m, None, 100)
        self.assertEqual(len(parts), 1)
        f, s, q = shape_fit.fit_score(shape_fit._biggest(m))
        self.assertGreater(f, 0.95)

    def test_two_touching_boxes_of_different_height_are_split(self):
        # L-shaped union: a big box with a smaller one next to it, bottoms aligned
        m = rect(240, 400, [(20, 40, 160, 160), (180, 100, 120, 100)])
        parts = shape_fit.split_blob(m, None, 200)
        self.assertEqual(len(parts), 2)

    def test_worker_like_blob_has_bad_fit(self):
        m = np.zeros((300, 300), np.uint8)
        cv2.circle(m, (150, 60), 35, 255, -1)                       # head
        cv2.rectangle(m, (110, 90), (190, 260), 255, -1)            # torso
        cv2.rectangle(m, (10, 120), (290, 150), 255, -1)            # arms stretched out
        f, s, q = shape_fit.fit_score(shape_fit._biggest(m))
        self.assertLess(f, 0.8)

    def test_quad_dims(self):
        q = np.float32([[0, 0], [100, 0], [100, 50], [0, 50]])
        L, S = shape_fit.quad_dims(q)
        self.assertAlmostEqual(L, 100, 3)
        self.assertAlmostEqual(S, 50, 3)


class TestSize(unittest.TestCase):
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], float)

    def setUp(self):
        # camera 100 cm above the belt, looking down at an angle (same convention as the pipeline)
        C = np.array([0.0, -100.0, 90.0])
        z = np.array([0.0, 0.0, 10.0]) - C
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        rv, _ = cv2.Rodrigues(R)
        self.cam = size_model.Camera3D(self.K, rv, (-R @ C).reshape(3, 1))

    def quad_px(self, cx, cy, L, D, h):
        P = [(cx - L / 2, cy - D / 2, h), (cx + L / 2, cy - D / 2, h), (cx + L / 2, cy + D / 2, h), (cx - L / 2, cy + D / 2, h)]
        return self.cam.project(P).astype(np.float32)

    def test_plane_scale_exact_for_same_height(self):
        ref = self.quad_px(-40, 0, 30, 20, 15)
        ps = size_model.PlaneScale.from_reference(ref, 30, 20)
        for cx, cy, L, D in ((0, 5, 40, 18), (50, -6, 25, 25)):
            m = ps.measure(self.quad_px(cx, cy, L, D, 15))
            self.assertAlmostEqual(m[0], max(L, D), delta=0.3)
            self.assertAlmostEqual(m[1], min(L, D), delta=0.3)

    def test_camera3d_recovers_size_and_height(self):
        ref = self.quad_px(-40, 0, 30, 20, 15)
        cam = size_model.Camera3D.from_reference(self.K, ref, 30, 20, 15)
        self.assertIsNotNone(cam)
        self.assertAlmostEqual(cam.C[2], 90.0, delta=0.5)
        # a taller box: top at 28 cm. Build the silhouette quad from its footprint + top corners
        L, D, h, cx, cy = 35, 22, 28, 20, 4
        top = self.quad_px(cx, cy, L, D, h)
        foot = self.quad_px(cx, cy, L, D, 0)
        sil = np.vstack([foot[:2], top[2:]])        # near corners from the belt, far corners from the top
        r = cam.measure(top, sil)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r[0], 35, delta=1.0)
        self.assertAlmostEqual(r[1], 22, delta=1.0)
        self.assertAlmostEqual(r[2], 28, delta=1.5)

    def test_estimator_rejects_outliers(self):
        e = size_model.SizeEstimator()
        for v in (30.1, 29.9, 30.0, 30.2, 29.8, 12.0, 61.0, 30.0):
            e.add(v, 20.0)
        r = e.result()
        self.assertAlmostEqual(r["L"], 30.0, delta=0.2)
        self.assertEqual(r["n"], 6)


if __name__ == "__main__":
    unittest.main()
