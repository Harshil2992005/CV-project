"""
shape_fit.py - "what is a box?" for the detector (fixes B1, B3, B4, B5).

The old detector only knew "a blob whose colour differs from the belt" and then drew a
minAreaRect around it. This file adds the missing knowledge about boxes:

  quad_from_contour   a box seen in perspective is a QUAD (parallelogram-like), not a
                      rectangle. We fit a 4-corner polygon to the convex hull instead of
                      minAreaRect, and measure width/height from opposite edges.
  fit_score           blob area / quad area. A real box fills its quad (~0.9+). A worker
                      or two touching boxes in an L-shape does not.
  split_blob          only when a blob does NOT fit one quad: try (1) convexity-defect
                      cuts, (2) strong straight image edges, (3) distance-transform
                      watershed. A cut is kept only if every piece fits a quad better.
  merge_pieces        pieces whose union fits one quad again are merged back (so a tape
                      line or a printed edge does not cut one box into two).
  edge_support        share of the outline that lies on a real image edge.
"""
import cv2
import numpy as np

FIT_OK = 0.90          # blob fills >= 90 % of its quad -> accept as ONE box without splitting
FIT_MIN = 0.72         # below this a piece is not a box at all
SOLID_MIN = 0.80
HEX_SOLIDITY = 0.95    # a box seen from the side (top+front+side face) is a convex hexagon:
HEX_FIT = 0.72         # convex enough = one box, never cut it between its faces


# ---------------------------------------------------------------- geometry
def order_quad(q):
    """tl, tr, br, bl (image coordinates)."""
    q = np.asarray(q, np.float32).reshape(4, 2)
    s, d = q.sum(1), np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def poly_area(p):
    p = np.asarray(p, np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def quad_from_contour(c):
    """Best 4-corner polygon of a contour (perspective aware). Returns ordered float32 (4,2)."""
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    best = None
    for e in np.linspace(0.004, 0.15, 24):
        ap = cv2.approxPolyDP(hull, e * peri, True)
        if len(ap) <= 4:
            best = ap
            break
    if best is not None and len(best) == 4:
        q = order_quad(best)
        # must stay roughly as big as the hull; otherwise it cut a corner too hard
        if poly_area(q) >= 0.92 * cv2.contourArea(hull):
            return q
    # fall back: smallest-area enclosing rectangle
    return order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))


def quad_dims(q):
    """(long side, short side) from the AVERAGE of opposite edges, in px."""
    a = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    b = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
    return (max(a, b), min(a, b))


def quad_ok(q):
    """Convex, no crazy angles (a perspective box has corners between ~45 and ~135 deg)."""
    q = np.asarray(q, np.float32)
    for i in range(4):
        v1, v2 = q[(i + 1) % 4] - q[i], q[(i - 1) % 4] - q[i]
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        if n < 1e-6:
            return False
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1)))
        if ang < 40 or ang > 140:
            return False
    return True


def fit_score(c):
    """(fit, solidity, quad). fit = blob area / quad area, 1.0 = perfect box silhouette."""
    area = cv2.contourArea(c)
    if area <= 0:
        return 0.0, 0.0, None
    q = quad_from_contour(c)
    qa = max(poly_area(q), 1e-6)
    hull_a = max(cv2.contourArea(cv2.convexHull(c)), 1e-6)
    return float(min(area / qa, 1.0)), float(area / hull_a), q


def edge_support(c, edges, tol=3):
    """Share of the outline pixels that have a strong image edge within `tol` px."""
    if edges is None:
        return 1.0
    h, w = edges.shape[:2]
    ol = np.zeros((h, w), np.uint8)
    cv2.drawContours(ol, [c], -1, 255, 1)
    near = cv2.dilate(edges, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8))
    n = int((ol > 0).sum())
    return float(((ol > 0) & (near > 0)).sum() / n) if n else 0.0


# ---------------------------------------------------------------- splitting
def _components(mask, min_area):
    n, cc, st, _ = cv2.connectedComponentsWithStats(mask)
    return [((cc == i).astype(np.uint8) * 255) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


def _biggest(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea) if cs else None


def _score_pieces(pieces):
    """Mean fit of the pieces, or -1 if any piece is not box-like."""
    fits = []
    for p in pieces:
        c = _biggest(p)
        if c is None:
            return -1.0
        f, s, _ = fit_score(c)
        if f < FIT_MIN or s < SOLID_MIN - 0.05:
            return -1.0
        fits.append(f)
    return float(np.mean(fits)) if fits else -1.0


def _cut_defects(comp, min_area):
    """Cut a blob at its convexity defects (the notch where two boxes meet)."""
    c = _biggest(comp)
    if c is None or len(c) < 5:
        return None
    hull = cv2.convexHull(c, returnPoints=False)
    try:
        defs = cv2.convexityDefects(c, hull)
    except cv2.error:
        return None
    if defs is None:
        return None
    thr = 0.06 * np.sqrt(cv2.contourArea(c))
    defs = np.asarray(defs).reshape(-1, 4)
    cc_ = c.reshape(-1, 2)
    D = sorted([(d[3] / 256.0, tuple(int(v) for v in cc_[d[2]]), tuple(int(v) for v in cc_[d[0]]),
                 tuple(int(v) for v in cc_[d[1]])) for d in defs if d[3] / 256.0 > thr], reverse=True)
    if not D:
        return None
    cut = comp.copy()
    far1 = D[0][1]
    if len(D) >= 2:                       # two notches facing each other: join them
        far2 = D[1][1]
        line = np.zeros_like(comp)
        cv2.line(line, far1, far2, 255, 1)
        inside = (line > 0).sum() and ((line > 0) & (comp > 0)).sum() / max((line > 0).sum(), 1) > 0.9
        if inside:
            cv2.line(cut, far1, far2, 0, 3)
            parts = _components(cut, min_area)
            if len(parts) >= 2:
                return parts
    # one notch: the two contour edges that meet in the notch are the faces of two boxes.
    # Continue either edge straight through the blob (that is where the boxes touch) and keep
    # the cut that leaves the best-fitting pieces; the bisector is the last resort.
    _, far, s, e = D[0]
    mid = ((s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0)
    dirs = []
    for v in ((far[0] - s[0], far[1] - s[1]), (far[0] - e[0], far[1] - e[1])):   # extend the edge
        n = np.hypot(*v)
        if n > 1e-3:
            dirs.append(np.array([v[0] / n, v[1] / n], np.float32))
    bis = np.array([far[0] - mid[0], far[1] - mid[1]], np.float32)
    if np.linalg.norm(bis) > 1e-3:
        dirs.append(bis / np.linalg.norm(bis))
    best = None
    for d in dirs:
        p = np.array(far, np.float32)
        cut = comp.copy()
        for _ in range(int(2 * max(comp.shape))):
            p = p + d
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < comp.shape[1] and 0 <= y < comp.shape[0]) or comp[y, x] == 0:
                break
            cv2.circle(cut, (x, y), 1, 0, -1)
        parts = _components(cut, min_area)
        if len(parts) >= 2:
            sc = _score_pieces(parts)
            if best is None or sc > best[0]:
                best = (sc, parts)
    return best[1] if best else None


def _cut_edges(comp, gray, min_area):
    """Cut a blob along a long straight image edge that runs from boundary to boundary."""
    h, w = comp.shape
    if gray is None:
        return None
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    ed = cv2.Canny(g, 40, 110)
    inner = cv2.erode(comp, np.ones((5, 5), np.uint8))
    ed &= inner
    minlen = int(0.35 * min(h, w))
    lines = cv2.HoughLinesP(ed, 1, np.pi / 180, threshold=max(15, minlen // 2),
                            minLineLength=max(minlen, 12), maxLineGap=6)
    if lines is None:
        return None
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
    best = None
    for l in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = [int(v) for v in l]
        # extend the segment both ways until it leaves the blob
        d = np.array([x2 - x1, y2 - y1], np.float32)
        L = np.linalg.norm(d)
        if L < 1:
            continue
        d /= L
        ends = []
        for sgn, p0 in ((1, (x2, y2)), (-1, (x1, y1))):
            p = np.array(p0, np.float32)
            for _ in range(int(max(h, w))):
                q = p + sgn * d
                xi, yi = int(round(q[0])), int(round(q[1]))
                if not (0 <= xi < w and 0 <= yi < h) or comp[yi, xi] == 0:
                    break
                p = q
            ends.append((int(round(p[0])), int(round(p[1]))))
        cut = comp.copy()
        cv2.line(cut, ends[0], ends[1], 0, 3)
        parts = _components(cut, min_area)
        if len(parts) >= 2:
            sc = _score_pieces(parts)
            if sc > 0 and (best is None or sc > best[0]):
                best = (sc, parts)
    return best[1] if best else None


def _cut_watershed(comp, img, min_area, peak_frac=0.5):
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    mx = dist.max()
    if mx <= 0:
        return None
    fg = (dist > peak_frac * mx).astype(np.uint8)
    n, mk = cv2.connectedComponents(fg)
    if n - 1 <= 1:
        return None
    markers = mk.astype(np.int32) + 1
    markers[(comp > 0) & (fg == 0)] = 0
    markers[comp == 0] = 1
    markers = cv2.watershed(img, markers)
    parts = [((markers == k).astype(np.uint8) * 255) for k in range(2, n + 1)]
    parts = [p for p in parts if p.sum() / 255 >= min_area]
    return parts if len(parts) >= 2 else None


def split_blob(comp, img, min_area, depth=0):
    """Split one blob into box-like pieces. Returns a list of masks (same size as comp).

    A blob that already fits one quad is NEVER split - so a tape line or a printed
    pattern cannot cut a box in half. Only blobs that are not box-shaped are cut.
    """
    c = _biggest(comp)
    if c is None:
        return []
    fit, sol, _ = fit_score(c)
    if sol >= HEX_SOLIDITY and fit >= HEX_FIT:
        return [comp]      # one clean quad, OR a convex 3-face (hexagon) silhouette of ONE box
    if depth >= 2:
        return [comp]
    base = fit if sol >= SOLID_MIN else fit * 0.8
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None and img.ndim == 3 else None
    best, best_sc = None, base + 0.03          # a split must clearly improve the fit
    for fn in (lambda: _cut_defects(comp, min_area),
               lambda: _cut_edges(comp, gray, min_area),
               lambda: _cut_watershed(comp, img, min_area) if img is not None else None):
        parts = fn()
        if not parts:
            continue
        sc = _score_pieces(parts)
        if sc > best_sc:
            best, best_sc = parts, sc
    if best is None:
        return [comp]
    out = []
    for p in best:                               # pieces may need another cut
        out += split_blob(p, img, min_area, depth + 1)
    return merge_pieces(out, min_area)


def merge_pieces(pieces, min_area):
    """Merge neighbouring pieces whose union is again one clean box (undo over-splitting)."""
    pieces = list(pieces)
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                grow = cv2.dilate(pieces[i], np.ones((7, 7), np.uint8))
                if not (grow & pieces[j]).any():
                    continue
                u = pieces[i] | pieces[j]
                u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                c = _biggest(u)
                if c is None:
                    continue
                f, s, _ = fit_score(c)
                if f >= FIT_OK + 0.02 and s >= 0.93:
                    pieces[i] = u
                    del pieces[j]
                    changed = True
                    break
            if changed:
                break
    return pieces


# ---------------------------------------------------------------- top face (fixes B3 / A3)
def top_face(reg, img):
    """The brightest, box-shaped part of a blob = the TOP face (light comes from above).

    A real box seen from the side shows its top, its front and often a side face. The
    silhouette of all three is NOT a rectangle in cm. The top face is: it is flat, and a
    known-size reference box gives its real scale. Returns an ordered quad in the SAME
    coordinates as `reg`, or None when no clear top face exists (blurred / evenly lit box).
    """
    c = _biggest(reg)
    if c is None or img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    inside = reg > 0
    vals = gray[inside]
    if vals.size < 200 or vals.std() < 6:
        return None
    t, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = ((gray > t) & inside).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # tape / labels
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    parts = _components(bright, 0.03 * cv2.contourArea(c))
    if not parts:
        return None
    # The top face is convex, and a tape band / label across it splits the "bright" area into
    # pieces. Take the hull of all bright pieces: the tape lies inside it, the darker front
    # and side faces lie outside it.
    pts = np.vstack([_biggest(m).reshape(-1, 2) for m in parts]).astype(np.int32)
    bc = cv2.convexHull(pts)
    frac = cv2.contourArea(bc) / max(cv2.contourArea(c), 1)
    fit, sol, q = fit_score(bc)
    if frac < 0.22 or frac > 0.88 or fit < 0.84 or not quad_ok(q):
        return None
    return q
