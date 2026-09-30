"""
Module B part 2 of 3 - build P = K [R | t] and take it apart again (RQ decomposition).

Why bother?
    The 3x4 matrix P does the whole projection in one go: 3-D point -> pixel.
    But we often want the pieces (how long is the lens, how is the board turned), so we
    split P back into K, R and t. That is an RQ decomposition, because
        P[:, :3] = K R
    and K is upper triangular while R is a rotation matrix.

    python moduleB/decompose_P.py moduleB/calibration.json     # real photo data
    python moduleB/decompose_P.py                              # synthetic example

Two ways to do the RQ step (see impl.py)
    base : scipy.linalg.rq                       - one line, somebody else tested it
    alt  : hand written, from QR plus row/column sign flips
           QR gives RQ when you flip the last rows and the last columns, because
           transposing a matrix swaps upper and lower triangular form.

Signs are fixed afterwards so diag(K) > 0 and K[2,2] = 1.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import impl                                            # noqa: E402


# ----------------------------------------------------------------------
# the two RQ implementations
# ----------------------------------------------------------------------

def rq_scipy(M):
    """ORIGINAL RQ decomposition, straight from SciPy."""
    from scipy.linalg import rq
    K, R = rq(M)
    return K, R


def rq_handmade(M):
    """ALTERNATIVE RQ decomposition, written by hand using only QR.

    We want  M = K Rot  with K upper triangular. QR cannot give that directly, but
    transposing swaps the two triangular shapes, and REVERSING all the rows and all
    the columns also swaps them. So:

        step 1   B = J M J                 (J = the matrix with 1s on the anti-diagonal)
                 Reversing rows and columns of an upper triangular matrix gives a
                 lower triangular one, so now B = K' Rot' with K' lower triangular.
        step 2   LQ of B, taken from the QR of B transposed.
                 If B^T = Qb Rb (QR, so Rb is upper triangular), then
                     B = (Rb^T) (Qb^T)        and Rb^T is lower triangular.
                 So  L = Rb^T  and  Ql = Qb^T.
        step 3   undo the reversal:  K = J L J  (upper triangular again)
                                 Rot = J Ql J (orthogonal again)
                 Because J J = I we get K Rot = J (L Ql) J = J B J = M, so the answer
                 really does multiply back to M.

    Finally we force det(Rot) = +1, the same convention SciPy uses.
    """
    n = M.shape[0]
    J = np.eye(n)[::-1]
    B = J @ M @ J
    Qb, Rb = np.linalg.qr(B.T)
    L, Ql = Rb.T, Qb.T
    K, Rot = J @ L @ J, J @ Ql @ J
    if np.linalg.det(Rot) < 0:                      # keep det(Rot) = +1, like SciPy
        E = np.diag([-1.0] + [1.0] * (n - 1))
        K, Rot = K @ E, E @ Rot
    return K, Rot


def rq3(M):
    """RQ decomposition, chosen by the impl mode."""
    return rq_handmade(M) if impl.is_alt() else rq_scipy(M)


def decompose(P):
    """Split the 3x4 projection matrix P into (K, R, t)."""
    K, R = rq3(P[:, :3])                 # the first 3 columns hide K and R
    # rq() leaves the signs free. Multiply K by T on the right and R by T on the left,
    # which changes nothing in K @ R but lets us force the signs we want.
    T = np.diag(np.sign(np.diag(K)))     # make every focal length positive
    K, R = K @ T, T @ R
    t = np.linalg.inv(K) @ P[:, 3]        # the last column divided by K
    s = K[2, 2]                           # normalise so K[2,2] is exactly 1
    return K / s, R, t / 1.0


# ----------------------------------------------------------------------
# self test + demo
# ----------------------------------------------------------------------

def self_test():
    """Build a P that we know the answer to, then check that decompose() gets it back.

    This is the point of doing this BEFORE trusting the code on real data: if the
    round trip is exact on a case where we know the answer, then a strange answer on
    real data means the input is strange, not the code.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        Kt = np.array([[rng.uniform(400, 1200), 0, 320],
                       [0, rng.uniform(400, 1200), 240],
                       [0, 0, 1.0]])
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        Rot, _ = cv2.Rodrigues(axis * rng.uniform(0.0, 1.2))
        tv = rng.uniform(-40, 40, 3)
        P = Kt @ np.hstack([Rot, tv.reshape(3, 1)])
        K2, R2, t2 = decompose(P)
        e = max(np.abs(Kt - K2).max(), np.abs(Rot - R2).max(), np.abs(tv - t2).max())
        worst = max(worst, float(e))
    print(f"RQ self-test over 20 random P matrices: worst error = {worst:.3e}")
    print("  (this is the code checking itself before it is used on real photos)\n")


def main():
    p = argparse.ArgumentParser(description="Split P = K [R|t] back into K, R and t.")
    p.add_argument("calib", nargs="?", default="", help="calibration.json, or nothing for a demo")
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    self_test()

    if a.calib:
        with open(a.calib) as f:
            c = json.load(f)
        K = np.array(c["K"])
        name, e = next(iter(c["extrinsics"].items()))     # take the first photo
        R, _ = cv2.Rodrigues(np.array(e["rvec"]))
        t = np.array(e["tvec"])
        print(f"Using extrinsics of {name}")
    else:
        K = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1]])
        R, _ = cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))
        t = np.array([10., -5., 300.])
        print("Synthetic example")

    P = K @ np.hstack([R, t.reshape(3, 1)])
    print("P = K [R|t] =\n", np.round(P, 3))
    K2, R2, t2 = decompose(P)
    print("\nRecovered K =\n", np.round(K2, 3))
    print("Recovered R =\n", np.round(R2, 4))
    print("Recovered t =", np.round(t2, 3))
    print(f"\nmax|K-K2|={np.abs(K - K2).max():.2e}  max|R-R2|={np.abs(R - R2).max():.2e}  "
          f"max|t-t2|={np.abs(t - t2).max():.2e}")
    print("det(R2) =", round(float(np.linalg.det(R2)), 6), "(should be +1)")

    # Cross-check against OpenCV's own answer.
    Kc, Rc, tc = cv2.decomposeProjectionMatrix(P)[:3]
    Kc = Kc / Kc[2, 2]
    print("OpenCV decomposeProjectionMatrix K matches:", np.allclose(Kc, K2, atol=1e-6))


if __name__ == "__main__":
    main()
