"""
Module E - What type of box is it?  (small / medium / large  ->  0 / 1 / 2)

Four classifiers, so we can compare them on the SAME crops:
  1. Eigenboxes  PCA eigenspace of the training crops + nearest neighbour
                 (this is the Eigenfaces idea applied to box crops)
  2. Alignment   ORB keypoints -> match -> RANSAC homography -> alignment residual
  3. Hu moments  7 rotation/scale/translation invariant numbers from the silhouette
  4. TypeBySize  the simplest one: just look at how big the crop is

Crops
  Either put your own pictures in  assets/crops/0/, assets/crops/1/, assets/crops/2/
  (one folder per type), or let the script cut them out of the video by itself. The
  video crops are labelled by clustering their sizes into 3 groups.

    python moduleE/recognize.py --video assets/video/sd_conveyor_2.mp4
    python moduleE/recognize.py --impl alt
Outputs: assets/eigenboxes_eigenvectors.png (mean + top-3 eigenboxes),
         assets/eigenboxes_accuracy.png and an accuracy table in the console.

Notes worth remembering for the report
  * LIGHTING: Alignment is fine with it, because keypoints only look at local
    edges, not at how bright things are. Eigenboxes store raw brightness, so a
    darker or brighter test picture moves it off the training manifold and hurts.
  * ROTATION: Hu moments are rotation invariant by construction, and the homography
    in Alignment absorbs rotation. Eigenboxes are NOT rotation invariant.
  * SIZE: Hu moments cannot see size, so they can only separate types that differ in
    shape or texture. That is why we keep the size classifier too.
  * These crops come from the SAME physical box seen in different frames, so the
    numbers are honest held-out splits across poses, not across different boxes.

Alternative functions (see impl.py)
    base : hand written 1-D k-means, NumPy SVD, median class size, cv2 remap
    alt  : sklearn KMeans, sklearn PCA, 1-NN on size, scikit-image adjust_gamma
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import belt_roi        # noqa: E402
import box_detect      # noqa: E402
import impl            # noqa: E402

CROP_DIR = os.path.join(ROOT, "assets", "crops")
TYPE_NAMES = {0: "small", 1: "medium", 2: "large", -1: "unknown"}
S = 64                                             # eigen-space image size


# ----------------------------------------------------------------------
# alternative helpers
# ----------------------------------------------------------------------

def change_light_multiply(img, gain):
    """ORIGINAL lighting change: just multiply every pixel."""
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def change_light_gamma(img, gain):
    """ALTERNATIVE lighting change: scikit-image gamma correction.

    gamma < 1 brightens, gamma > 1 darkens, and it touches the dark parts much more
    than the bright parts, which is what a real light change looks like.
    """
    from skimage.exposure import adjust_gamma
    out = adjust_gamma(img.astype(np.float32) / 255.0, gamma=gain)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def change_light(img, gain):
    return change_light_gamma(img, gain) if impl.is_alt() else change_light_multiply(img, gain)


# ----------------------------------------------------------------------
# crop collection / loading
# ----------------------------------------------------------------------

def crop_box(frame, b, pad=0.12):
    """Cut out a detected box, plus a little belt around it.

    The margin matters: the classifiers also look at the silhouette and at the belt
    colour, so the crop has to contain some belt. pad=0.12 means 12 % extra on each side.
    """
    H, W = frame.shape[:2]
    px, py = int(b["w"] * pad), int(b["h"] * pad)
    x1, y1 = max(b["x"] - px, 0), max(b["y"] - py, 0)
    x2, y2 = min(b["x"] + b["w"] + px, W), min(b["y"] + b["h"] + py, H)
    return frame[y1:y2, x1:x2]


def kmeans_1d_numpy(v, k, iters=50):
    """ORIGINAL: 1-D k-means written by hand."""
    v = np.asarray(v, float)
    # Start the k centres at evenly spaced percentiles of the data.
    c = np.quantile(v, np.linspace(0.1, 0.9, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(v[:, None] - c[None]), axis=1)     # nearest centre
        new = np.array([v[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    # Renumber so the smallest size always ends up as class 0.
    order = np.argsort(c)
    remap = {int(o): i for i, o in enumerate(order)}
    return np.array([remap[int(l)] for l in lab])


def kmeans_1d_sklearn(v, k, iters=50):
    """ALTERNATIVE: 1-D k-means from scikit-learn (same job, tested implementation)."""
    from sklearn.cluster import KMeans
    v = np.asarray(v, float).reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=10, max_iter=iters, random_state=0).fit(v)
    lab = km.labels_
    c = km.cluster_centers_.ravel()
    order = np.argsort(c)
    remap = {int(o): i for i, o in enumerate(order)}
    return np.array([remap[int(l)] for l in lab])


def kmeans_1d(v, k, iters=50):
    """Cluster numbers into k groups, chosen by the impl mode."""
    return kmeans_1d_sklearn(v, k, iters) if impl.is_alt() else kmeans_1d_numpy(v, k, iters)


def collect_crops(video, roi=None, n_types=3, per_class=15, save=True):
    """Cut clean (not border-cut) boxes out of the video and label them by size cluster."""
    if roi is None:
        roi = belt_roi.get_roi(video)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    items = []                                     # (crop, size)
    step = max(1, n // 150)                        # look at about 150 frames
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        boxes, _ = box_detect.segment_boxes(frame, roi)
        for b in boxes:
            if b["touches_border"]:
                continue                          # a cut-off box has no true size
            crop = crop_box(frame, b)
            if crop.size:
                items.append((crop.copy(), np.sqrt(b["rw"] * b["rh"])))
    cap.release()
    if len(items) < n_types * 3:
        raise SystemExit("[!] Too few boxes found to build crops. Check the belt ROI / detector.")
    labels = kmeans_1d([s for _, s in items], n_types)

    imgs, labs = [], []
    for lab in range(n_types):
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        pick = idx[np.linspace(0, len(idx) - 1, min(per_class, len(idx))).astype(int)]
        for j in pick:
            imgs.append(items[j][0])
            labs.append(lab)

    if save:
        # Start fresh, so old crops never mix with the new ones.
        for lab in set(labs):
            os.makedirs(os.path.join(CROP_DIR, str(lab)), exist_ok=True)
            for old in glob.glob(os.path.join(CROP_DIR, str(lab), "*.png")):
                os.remove(old)
        cnt = {}
        for im, lab in zip(imgs, labs):
            cnt[lab] = cnt.get(lab, 0) + 1
            cv2.imwrite(os.path.join(CROP_DIR, str(lab), f"{cnt[lab]:03d}.png"), im)
        print(f"[+] saved {len(imgs)} crops in {CROP_DIR}: " + str({k: v for k, v in sorted(cnt.items())}))
    return imgs, np.array(labs)


def load_crops():
    """Read (images, labels) from assets/crops/<label>/*.png. None if there are too few."""
    imgs, labs = [], []
    for d in sorted(glob.glob(os.path.join(CROP_DIR, "[0-9]*"))):
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))):
            im = cv2.imread(p)
            if im is not None:
                imgs.append(im)
                labs.append(int(os.path.basename(d)))
    if len(set(labs)) < 2:
        return None
    return imgs, np.array(labs)


# ----------------------------------------------------------------------
# 4. size classifier (the one the pipeline uses)
# ----------------------------------------------------------------------

class TypeBySize:
    """Nearest class size. base = median class size, alt = 1-NN on the size number."""

    def build(self, images, labels):
        labels = np.asarray(labels)
        s = np.array([np.sqrt(im.shape[0] * im.shape[1]) for im in images])
        self.classes = sorted(set(labels.tolist()))
        if impl.is_alt():
            from sklearn.neighbors import KNeighborsClassifier
            self.knn = KNeighborsClassifier(n_neighbors=1).fit(s.reshape(-1, 1), labels)
            self.centres = None
        else:
            self.knn = None
            self.centres = np.array([np.median(s[labels == c]) for c in self.classes])
        return self

    def predict(self, crop):
        if crop is None or crop.size == 0:
            return -1
        s = np.sqrt(crop.shape[0] * crop.shape[1])
        if self.knn is not None:
            return int(self.knn.predict(np.array([[s]]))[0])
        return int(self.classes[int(np.argmin(np.abs(self.centres - s)))])


# ----------------------------------------------------------------------
# 1. Eigenboxes
# ----------------------------------------------------------------------

def _vec(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0


class Eigenboxes:
    """ORIGINAL = SVD, ALTERNATIVE = sklearn PCA.

    Both find the same directions of biggest variation. With PCA the sign of each
    component can come out flipped compared to SVD, so the saved eigenbox pictures can
    look inverted. That does not change the classification at all.
    """

    def __init__(self, n_components=10):
        self.k = n_components

    def build(self, images, labels):
        X = np.array([_vec(i) for i in images])
        self.mean = X.mean(0)
        if impl.is_alt():
            from sklearn.decomposition import PCA
            p = PCA(n_components=self.k, svd_solver="full").fit(X - self.mean)
            self.basis = np.asarray(p.components_)
            self.k = len(self.basis)
            self.proj = p.transform(X - self.mean)
        else:
            U, Sv, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
            self.k = min(self.k, len(Vt))
            self.basis = Vt[:self.k]                   # eigenboxes (rows)
            self.proj = (X - self.mean) @ self.basis.T
        self.labels = np.asarray(labels)
        return self

    def predict(self, img):
        z = (_vec(img) - self.mean) @ self.basis.T
        return int(self.labels[np.argmin(np.linalg.norm(self.proj - z, axis=1))])

    def save_top3(self, path):
        tiles = []
        for e in self.basis[:3]:
            e = e.reshape(S, S)
            tiles.append(cv2.resize(((e - e.min()) / (np.ptp(e) + 1e-9) * 255).astype(np.uint8),
                                    (192, 192), interpolation=cv2.INTER_NEAREST))
        m = ((self.mean.reshape(S, S)) * 255).astype(np.uint8)
        tiles.insert(0, cv2.resize(m, (192, 192)))
        cv2.imwrite(path, np.hstack(tiles))


# ----------------------------------------------------------------------
# 2. Alignment (ORB + homography + residual)
# ----------------------------------------------------------------------

class Alignment:
    """ORIGINAL = cv2.perspectiveTransform, ALTERNATIVE = manual homogeneous multiply."""

    def __init__(self, size=128, per_class=6):
        self.size, self.per_class = size, per_class
        self.orb = cv2.ORB_create(400, fastThreshold=5, edgeThreshold=8, patchSize=15)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _feat(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # 16 px border, so features near the edge are not cut off by the resize.
        g = cv2.copyMakeBorder(cv2.resize(g, (self.size, self.size)), 16, 16, 16, 16, cv2.BORDER_REPLICATE)
        g = cv2.createCLAHE(2.0, (4, 4)).apply(g)      # equalise contrast locally
        return self.orb.detectAndCompute(g, None)

    def build(self, images, labels):
        self.templates = []
        labels = np.asarray(labels)
        for c in sorted(set(labels.tolist())):
            idx = np.where(labels == c)[0]
            for j in idx[np.linspace(0, len(idx) - 1, min(self.per_class, len(idx))).astype(int)]:
                kp, des = self._feat(images[j])
                if des is not None and len(kp) >= 4:
                    self.templates.append((c, kp, des))
        return self

    def score(self, kq, dq, kt, dt):
        """Lower is better. RANSAC homography residual in px, plus a penalty for few inliers."""
        m = self.bf.match(dq, dt)
        if len(m) < 4:
            return np.inf
        src = np.float32([kq[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        dst = np.float32([kt[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hm is None or inl.sum() < 4:
            return np.inf
        if impl.is_alt():
            # ALTERNATIVE: project the points by hand.
            #   [u' v' w'] = H @ [x y 1]',   then u = u'/w',  v = v'/w'
            ones = np.ones((src.shape[0], 1), np.float32)
            hom = np.hstack([src.reshape(-1, 2), ones])       # N x 3
            proj = (Hm @ hom.T).T                             # N x 3
            proj = proj[:, :2] / proj[:, 2:3]
            proj = proj.reshape(-1, 1, 2)
        else:
            # ORIGINAL: OpenCV does the same division for us.
            proj = cv2.perspectiveTransform(src, Hm)
        res = np.linalg.norm(proj - dst, axis=2).ravel()[inl.ravel() == 1]
        return float(res.mean() + 20.0 / inl.sum())

    def predict(self, img):
        kq, dq = self._feat(img)
        if dq is None or len(kq) < 4:
            return -1
        best, lab = np.inf, -1
        for c, kt, dt in self.templates:
            s = self.score(kq, dq, kt, dt)
            if s < best:
                best, lab = s, c
        return int(lab)


# ----------------------------------------------------------------------
# 3. Hu moments
# ----------------------------------------------------------------------

def hu_features(img):
    """Log Hu moments of (a) the silhouette and (b) the brightness-normalised masked image."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sc = (S * 2) / max(g.shape)                      # ISOTROPIC resize (keep aspect ratio!)
    g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    # silhouette = pixels that differ from the crop border colour (= belt)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    diff = np.abs(g.astype(np.int16) - int(np.median(border))).astype(np.uint8)
    t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (diff > max(t, 8)).astype(np.uint8)
    hs = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).ravel()
    inside = g.astype(np.float32) * mask
    if mask.sum() > 0:
        inside /= (inside.sum() / mask.sum() + 1e-6)          # remove overall brightness
    hg = cv2.HuMoments(cv2.moments(inside)).ravel()
    # h1, h2 are stable; higher Hu moments of near-symmetric boxes are ~0 and only add noise
    f = np.concatenate([hs[:2], hg[:2]])
    return -np.sign(f) * np.log10(np.abs(f) + 1e-30)          # -sign(h)*log10(|h|)


class HuClassifier:
    def build(self, images, labels):
        F = np.array([hu_features(i) for i in images])
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        Z = (F - self.mu) / self.sd
        labels = np.asarray(labels)
        self.classes = sorted(set(labels.tolist()))
        self.centres = np.array([Z[labels == c].mean(0) for c in self.classes])
        return self

    def predict(self, img):
        z = (hu_features(img) - self.mu) / self.sd
        return int(self.classes[int(np.argmin(np.linalg.norm(self.centres - z, axis=1)))])


# ----------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------

def rotate_crop(img, angle):
    """Rotate a crop on a bigger canvas filled with the belt colour (median of the border)."""
    h, w = img.shape[:2]
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).reshape(-1, img.shape[-1] if img.ndim == 3 else 1)
    fill = tuple(int(v) for v in np.median(edge, axis=0))
    d = int(np.ceil(np.hypot(h, w))) + 4
    big = np.full((d, d) + ((3,) if img.ndim == 3 else ()), fill if img.ndim == 3 else fill[0], np.uint8)
    y0, x0 = (d - h) // 2, (d - w) // 2
    big[y0:y0 + h, x0:x0 + w] = img
    M = cv2.getRotationMatrix2D((d / 2, d / 2), angle, 1.0)
    return cv2.warpAffine(big, M, (d, d), borderValue=fill if img.ndim == 3 else fill[0])


def accuracy(clf, imgs, labels):
    if not len(imgs):
        return 0.0
    return float(np.mean([clf.predict(i) == l for i, l in zip(imgs, labels)]))


def split(images, labels, test_frac=0.3, seed=0):
    """Split into train / test INSIDE every class, so both parts have all types."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in sorted(set(labels.tolist())):
        idx = rng.permutation(np.where(labels == c)[0])
        k = max(1, int(round(len(idx) * test_frac)))
        te += idx[:k].tolist()
        tr += idx[k:].tolist()
    return tr, te


def main():
    p = argparse.ArgumentParser(description="Train and compare four box-type classifiers.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--recollect", action="store_true", help="cut new crops from the video")
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    video = os.path.join(ROOT, a.video)
    print(f"[moduleE] impl={impl.get_mode()}")
    data = None if a.recollect else load_crops()
    if data is None:
        data = collect_crops(video, roi=belt_roi.get_roi(video))
    images, labels = data
    print("samples per type:", {int(c): int((labels == c).sum()) for c in sorted(set(labels.tolist()))})

    tr, te = split(images, labels)
    Xtr, ytr = [images[i] for i in tr], labels[tr]
    Xte, yte = [images[i] for i in te], labels[te]

    clfs = {"Eigenboxes": Eigenboxes().build(Xtr, ytr),
            "Alignment": Alignment().build(Xtr, ytr),
            "Hu moments": HuClassifier().build(Xtr, ytr),
            "Size-based": TypeBySize().build(Xtr, ytr)}
    clfs["Eigenboxes"].save_top3(os.path.join(ROOT, "assets", "eigenboxes_eigenvectors.png"))

    # Three "hard" versions of the test set, one per thing a real system must survive.
    rng = np.random.default_rng(5)
    Xrot = [rotate_crop(i, float(rng.uniform(20, 160))) for i in Xte]
    Xdark = [change_light(i, 0.6) for i in Xte]
    Xbright = [change_light(i, 1.4) for i in Xte]

    rows = {}
    print(f"\n{'classifier':<12}{'held-out':>10}{'rotated':>10}{'darker':>10}{'brighter':>10}")
    for name, clf in clfs.items():
        r = [accuracy(clf, Xte, yte), accuracy(clf, Xrot, yte),
             accuracy(clf, Xdark, yte), accuracy(clf, Xbright, yte)]
        rows[name] = r
        print(f"{name:<12}" + "".join(f"{v * 100:>9.0f}%" for v in r))

    # Show the invariance of the Hu features: how far they move under a 37 deg turn.
    d = np.mean([np.linalg.norm(hu_features(i) - hu_features(rotate_crop(i, 37))) for i in Xte])
    print(f"\nHu feature change after a 37 deg rotation (mean L2, log scale): {d:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.2
    for k, (lab) in enumerate(["held-out", "rotated", "darker", "brighter"]):
        ax.bar(np.arange(len(rows)) + k * w, [v[k] * 100 for v in rows.values()], w, label=lab)
    ax.set_xticks(np.arange(len(rows)) + 1.5 * w)
    ax.set_xticklabels(rows.keys())
    ax.set_ylabel("accuracy %"); ax.set_ylim(0, 105); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "eigenboxes_accuracy.png"), dpi=120)
    print("[+] assets/eigenboxes_eigenvectors.png, assets/eigenboxes_accuracy.png")


if __name__ == "__main__":
    main()
