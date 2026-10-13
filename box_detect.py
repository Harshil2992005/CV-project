"""
box_detect.py - the ONE box detector that pipeline, live_demo, module C/D/E all use.

Why there is only one detector
    If every script had its own detection code, the pictures in module C and the boxes
    in the pipeline would not match. So everybody calls segment_boxes() from here.

How a box is found, step by step
    1. CUT the frame down to the belt rectangle (see belt_roi.py). We crop instead of
       painting the outside black, because black areas used to look like dark boxes.
    2. LEARN THE BELT COLOUR by taking the median colour of the whole belt area.
    3. A pixel is "not belt" when its LAB colour is more than MIN_COLOR_DIFF away from
       that median. This works for dark, blue, green or printed boxes, because nothing
       about brightness is hard coded.
    4. Clean the mask: open (remove specks), close (join gaps), fill holes.
       Filling holes matters because a sticker or a barcode inside a box would
       otherwise become a hole and then a second "box".
    5. Two boxes that touch get split apart with a distance transform + watershed.
    6. Every blob becomes a ROTATED rectangle (minAreaRect), so a tilted box still
       gets a correct width and height.
    7. All the size limits are given as a FRACTION of the belt area, so the same code
       works for both the SD and the HD videos.
8. A box that is cut by the left or right edge of the belt gets the flag
    "touches_border". Its size is not trustworthy, so do not measure it.
9. If a BackgroundFilter is given, a detection that looks exactly like the still
    room behind the belt AND has not moved is thrown away, because a wall panel or a
    floor tile is a rectangle too and used to be reported as a box.


Alternative functions (see impl.py)
    base : cv2.threshold + cv2.drawContours + cv2.morphologyEx + cv2.boxPoints
    alt  : skimage threshold_otsu + floodFill hole filling + erode/dilate + own corners
    Run with  --impl alt  to use the second row. The results are the same, only the
    library behind each step is different.

What segment_boxes() gives back (a list of dictionaries)
    x, y, w, h    plain axis-aligned bounding box, in full-video pixels
    cx, cy        centre point of the box
    rw, rh        size of the ROTATED rectangle, rw is always >= rh
    pts           the 4 corner points of the rotated rectangle
    touches_border  True when the box is cut off by the side of the belt
"""
import argparse
import os

import cv2
import numpy as np

import belt_roi
import impl
import shape_fit
import scene_check

ROOT = os.path.dirname(os.path.abspath(__file__))

# Two tuning numbers. Both are "relative", not fixed pixel counts.
MIN_COLOR_DIFF = 12.0        # LAB distance below this counts as "same colour as belt"
MIN_RECTANGULARITY = 0.50    # blob area / rotated-rectangle area; lower = more lenient

# Box-shape knowledge (shape_fit.py)
REJECT = {}                 # debug: why blobs were thrown away
MIN_FIT = 0.72              # blob area / quad area
MIN_SOLIDITY = 0.80         # blob area / convex hull area
MIN_EDGE_SUPPORT = 0.12     # share of the outline lying on a real image edge
MAX_HIVIS_FRAC = 0.35       # yellow-green high-vis vest pixels -> a worker, not a box
MAX_BLUE_FRAC = 0.50        # blue bins / gloves / belt side wall (only if not box-shaped)

# Two more tuning numbers, used by BackgroundFilter.
STATIC_BG_MATCH = 0.90       # this much of the blob looking like the still room = not a box
STATIC_MOVE_PX = 4.0         # ... and it must not have moved further than this, in px


# ----------------------------------------------------------------------
# small alternative helpers
# ----------------------------------------------------------------------

def otsu_cv2(d8):
    """ORIGINAL threshold: let OpenCV pick the split value with its Otsu method."""
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def otsu_skimage(d8):
    """ALTERNATIVE threshold: scikit-image picks the split value.

    Same idea (Otsu picks the threshold that makes the two classes as different as
    possible) but written by scikit-image. It returns a float, so we round it to match
    what OpenCV gives us.
    """
    from skimage.filters import threshold_otsu
    return float(round(threshold_otsu(d8)))


def otsu_threshold(d8):
    """Pick the Otsu threshold using whichever library the impl mode asks for."""
    return otsu_skimage(d8) if impl.is_alt() else otsu_cv2(d8)


def morph_cv2(mask, k):
    """ORIGINAL cleanup: one open followed by one close, using morphologicalEx."""
    se = np.ones((k, k), np.uint8)                     # square structuring element
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)   # shrinks then grows = kills specks
    se2 = np.ones((k + 4, k + 4), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)  # grows then shrinks = closes gaps


def morph_manual(mask, k):
    """ALTERNATIVE cleanup: the same erode/dilate calls written out one by one.

    OPEN   = erode then dilate
    CLOSE  = dilate then erode
    Doing it by hand shows what morphologicalEx actually does underneath.
    """
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))    # remove small white bits
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))   # bring back the real blobs
    mask = cv2.dilate(mask, np.ones((k + 4, k + 4), np.uint8))   # plug small holes
    return cv2.erode(mask, np.ones((k + 4, k + 4), np.uint8))   # restore the outer size


def clean_mask(mask, k):
    """Erode/dilate cleanup, chosen by the impl mode."""
    return morph_manual(mask, k) if impl.is_alt() else morph_cv2(mask, k)


def fill_holes_cv2(mask):
    """ORIGINAL hole filling: take the outer contour of everything and paint it in."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, -1)         # thickness -1 = filled
    return filled


def fill_holes_floodfill(mask):
    """ALTERNATIVE hole filling: flood the background inwards from the border.

    Idea: start a flood fill on the outside of the picture with a new value (say 1).
    Everything the water can reach is real background. Whatever stays at the old value
    (0) inside the picture is a hole, so we turn it white. No contour finder needed.
    """
    h, w = mask.shape[:2]
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    # floodFill needs a mask that is 2 pixels bigger than the image and starts at 0.
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, ff, (0, 0), 1)
    holes = (pad[1:-1, 1:-1] == 0)                       # 0 after the flood = hole
    out = mask.copy()
    out[holes] = 255
    return out


def fill_holes(mask):
    """Fill the holes of a mask, chosen by the impl mode."""
    return fill_holes_floodfill(mask) if impl.is_alt() else fill_holes_cv2(mask)


def rect_corners_cv2(rect):
    """ORIGINAL: ask OpenCV for the 4 corners of the rotated rectangle."""
    return cv2.boxPoints(rect).astype(int)


def rect_corners_manual(rect):
    """ALTERNATIVE: compute the 4 corners ourselves from the centre, size and angle.

    minAreaRect gives us ((cx, cy), (w, h), angle). The rectangle has half-width a and
    half-height b, and we rotate the axes by the angle. The 4 corners are just
    centre +/- a*u +/- b*v where u and v are the two rotated axis vectors.
    """
    (cx, cy), (w, h), angle = rect
    a, b = w / 2.0, h / 2.0
    rad = np.deg2rad(angle)
    ux, uy = np.cos(rad), np.sin(rad)      # direction of the first axis
    vx, vy = -np.sin(rad), np.cos(rad)     # direction of the second axis (90 deg)
    pts = []
    for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        x = cx + su * a * ux + sv * b * vx
        y = cy + su * a * uy + sv * b * vy
        pts.append((x, y))
    return np.array(pts, np.int32).astype(int)


def rect_corners(rect):
    """The 4 corners of the rotated rectangle, chosen by the impl mode."""
    return rect_corners_manual(rect) if impl.is_alt() else rect_corners_cv2(rect)


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------

def iou(a, b):
    """Intersection over Union of two plain bounding boxes (how much they overlap)."""
    x1, y1 = max(a["x"], b["x"]), max(a["y"], b["y"])            # overlap start
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])                   # overlap end
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)                     # overlap area
    if inter == 0:
        return 0.0                                               # no overlap at all
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)  # over the union


class BackgroundFilter:
    """Throw away detections that are the room behind the belt, not a box on it.

    Why this file needs one more class
        The detector only knows "a rectangle whose colour differs from the median
        colour of the belt area". A wall panel, a poster or a floor tile is a
        rectangle as well, so with a wide belt area they were all reported as boxes.
        Two facts tell the two cases apart:
            1. a real box is carried along by the belt, a wall panel never moves, and
            2. a wall panel looks exactly like the scene from before the box arrived.
        So a detection is only thrown away when BOTH hold: it matches the still
        background, and it stands in the same place as the accepted detection from
        the previous frame. A box that happens to look like the background but is
        travelling is always kept.

    The background picture
        A plain running average would slowly swallow a box that stands still on the
        belt and then throw that box away. So the background is only updated on the
        pixels that are NOT covered by an accepted box, and very slowly (alpha).
        It is refreshed every few frames instead of every frame, because it moves at
        0.2 % per step and the blur + colour conversion of a full HD picture is the
        most expensive part of this class.

        filter(frame, boxes) -> (kept_boxes, dropped_so_far)
    """

    def __init__(self, alpha=0.002, match=STATIC_BG_MATCH, move_px=STATIC_MOVE_PX,
                 every=4, memory=3):
        self.alpha = alpha                  # how fast the background follows the scene
        self.match = match                  # share of the blob that must look still
        self.move_px = move_px              # how far a "standing still" box may drift
        self.every = every                  # refresh the background every N frames
        self.memory = memory                # how many past frames count as "still here"
        self.bg = None                      # the background picture, float32 BGR
        self.bg_lab = None                  # the same picture in LAB, for the comparison
        self.prev = []                      # candidates of the last few frames
        self.dropped = 0                    # running total of thrown-away detections
        self.tick = 0

    @staticmethod
    def _lab(img):
        """Blurred LAB picture, float32, ready to be compared pixel by pixel.

        The picture is always handed to OpenCV as uint8 first. That matters: BGR2LAB
        uses a different scale for float input (it then expects 0..1), so mixing a
        uint8 frame with a float32 background would compare two different scales and
        every pixel would look different.
        """
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0),
                            cv2.COLOR_BGR2LAB).astype(np.float32)

    def _is_standing_still(self, b):
        """True when b sits on top of a candidate seen in one of the last few frames.

        More than one frame is remembered because the detector blinks: a wall panel is
        found in roughly every second frame, so comparing with the previous frame only
        would let it through every second time.
        """
        return any(iou(b, p) >= 0.5
                   and abs(b["cx"] - p["cx"]) + abs(b["cy"] - p["cy"]) < self.move_px
                   for past in self.prev for p in past)

    def filter(self, frame, boxes):
        """Return (kept_boxes, dropped_so_far) for one frame.

        `prev` keeps the candidates of the last few frames, dropped ones included.
        The list is only used to answer "is something standing here", so a wall panel
        that is thrown away must still be remembered, otherwise it would be accepted
        again on the next frame, dropped again on the one after, and so on.
        """
        H, W = frame.shape[:2]
        if self.bg is None:                 # first frame: there is nothing to compare with
            self.bg = frame.astype(np.float32)
            self.bg_lab = self._lab(self.bg)
            self.prev = [list(boxes)]
            return list(boxes), 0

        keep = np.ones((H, W), bool)
        for b in boxes:                     # never learn from pixels a box is sitting on
            x0, y0 = max(int(b["x"]), 0), max(int(b["y"]), 0)
            keep[y0:int(b["y"] + b["h"]), x0:int(b["x"] + b["w"])] = False

        lab = self._lab(frame)
        kept = []
        for b in boxes:
            x0, y0 = max(int(b["x"]), 0), max(int(b["y"]), 0)
            x1, y1 = min(int(b["x"] + b["w"]), W), min(int(b["y"] + b["h"]), H)
            if x1 - x0 < 4 or y1 - y0 < 4:
                kept.append(b)             # too small to judge, so keep it
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [b["pts"] - np.array([x0, y0])], 255)
            inside = m > 0
            if not inside.any():
                inside[:] = True            # the corners fell outside, use the whole patch
            d = np.linalg.norm(lab[y0:y1, x0:x1] - self.bg_lab[y0:y1, x0:x1], axis=2)
            same = float((d[inside] < MIN_COLOR_DIFF).mean())
            if same >= self.match and self._is_standing_still(b):
                self.dropped += 1
                continue
            kept.append(b)

        self.prev.append(list(boxes))
        if len(self.prev) > self.memory:
            self.prev.pop(0)
        self.tick += 1
        if self.tick % self.every == 0:     # slow refresh, alpha scaled to match
            k = self.alpha * self.every
            self.bg += k * (frame.astype(np.float32) - self.bg) * keep[:, :, None]
            self.bg_lab = self._lab(self.bg)
        return kept, self.dropped


def _split_touching(comp, img, peak_frac=0.45):
    """Cut one blob into several masks if it hides more than one box.

    The distance transform gives, for every pixel, the distance to the nearest
    background pixel. Deep inside a box that distance is big, in the thin neck between
    two touching boxes it is small. So every box gets one "peak". We seed the watershed
    with those peaks and let the water rise, which splits the blob at the neck.
    """
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    mx = dist.max()
    if mx <= 0:
        return [comp]                          # empty mask, nothing to split
    fg = (dist > peak_frac * mx).astype(np.uint8)   # only the deep centres stay
    n, mk = cv2.connectedComponents(fg)             # how many separate centres?
    if n - 1 <= 1:
        return [comp]                          # only one centre = one box
    markers = mk.astype(np.int32) + 1          # background -> 1, seeds -> 2 .. n
    markers[(comp > 0) & (fg == 0)] = 0        # the "not sure yet" zone
    markers[comp == 0] = 1                     # definitely background
    markers = cv2.watershed(img, markers)      # flood from the seeds
    return [((markers == k).astype(np.uint8) * 255) for k in range(2, n + 1)]


def _context_frac(hsv, mask):
    """Share of the blob that is hi-vis yellow-green or blue (workers, bins, belt wall)."""
    m = mask > 0
    n = max(int(m.sum()), 1)
    h, sat, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hivis = (h >= 22) & (h <= 45) & (sat > 110) & (v > 140)
    blue = (h >= 100) & (h <= 130) & (sat > 90) & (v > 40)
    return {"hivis": float((hivis & m).sum() / n), "blue": float((blue & m).sum() / n)}


def _belt_region(lab, belt_lab, tol=5.0):
    """Per-frame belt surface (convex hull), for footage where the camera moves.

    The belt has a known grey-green colour (roi["belt_lab"], sampled once from an empty
    stretch). Pixels near that colour are belt; the largest connected piece, closed over
    the boxes that sit on it and wrapped in a convex hull, is the belt region. A fixed
    rectangle/polygon cannot follow a hand-held camera, this does.
    """
    h, w = lab.shape[:2]
    sc = 4                                          # work at 1/4 size, it is only a region
    small = cv2.resize(lab, (w // sc, h // sc), interpolation=cv2.INTER_AREA)
    # Chroma only (a,b): the belt gets lighter and darker with shadow, its hue does not.
    ab = np.linalg.norm(small[:, :, 1:] - np.array(belt_lab[1:], np.float32), axis=2)
    m = ((ab < tol) & (small[:, :, 0] > 50) & (small[:, :, 0] < 150)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.full((h, w), 255, np.uint8)       # no belt found: do not hide anything
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    hull = cv2.convexHull(cv2.findNonZero((cc == k).astype(np.uint8)))
    reg = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(reg, hull, 255)
    # Tall boxes stick up above the belt's far edge, so also take a band above the hull.
    up = int(0.10 * m.shape[0])
    cv2.fillConvexPoly(reg, np.maximum(hull - np.array([0, up]), 0), 255)
    reg = cv2.morphologyEx(reg, cv2.MORPH_CLOSE, np.ones((up * 2 + 1, 1), np.uint8))
    return cv2.resize(reg, (w, h), interpolation=cv2.INTER_NEAREST)


def _cardboard_mask(work, roi, ox, oy):
    """Mask of cardboard-coloured pixels that lie inside the belt polygon.

    Used when belt_roi.json carries hsv_lo / hsv_hi (and optionally poly). Colour alone
    cannot tell a box from a worker or a wall in a busy scene, so we only accept pixels
    that are (a) cardboard coloured and (b) on the belt surface. The tall closing kernel
    glues the top face and the front face of one box together across the tape/label line.
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(work, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(roi["hsv_lo"], np.uint8), np.array(roi["hsv_hi"], np.uint8))
    if "poly" in roi:
        pm = np.zeros(m.shape, np.uint8)
        cv2.fillPoly(pm, [np.array(roi["poly"], np.int32) - np.array([ox, oy])], 255)
        m &= pm
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 9), np.uint8))


# ----------------------------------------------------------------------
# the detector itself
# ----------------------------------------------------------------------

def segment_boxes(frame, roi=None, min_area_frac=0.004, max_area_frac=0.6, static=None, plate=None):
    """Find every box in one video frame.

    frame          the colour frame (BGR)
    roi            belt rectangle; None means "use the whole frame"
    min_area_frac  smallest allowed box, as a fraction of the belt area
    max_area_frac  biggest allowed box, as a fraction of the belt area
    static         a BackgroundFilter that throws away still room, or None
    plate          FIXED camera only: the empty-scene picture (scene_check.build_plate). The
                   detector then finds "what differs from the empty scene" - no belt colour
                   and no wall/panel false boxes. Same size as the frame.
    returns        (boxes, picture_with_boxes_drawn_on_it)
    """
    H, W = frame.shape[:2]
    if roi is None:
        roi = {"x": 0, "y": 0, "w": W, "h": H}

    # Crop to the belt. ox/oy is where the crop starts inside the full frame.
    ox, oy = max(int(roi["x"]), 0), max(int(roi["y"]), 0)
    work = frame[oy:oy + int(roi["h"]), ox:ox + int(roi["w"])]
    h, w = work.shape[:2]
    if h < 10 or w < 10:
        return [], frame.copy()                 # crop too small, nothing to do

    # The picture we draw on is a copy of the full frame, so boxes sit on the video.
    vis = frame.copy()
    cv2.rectangle(vis, (ox, oy), (ox + w, oy + h), (0, 255, 255), 2)

    # --- step 1: what colour is the belt? ---
    # LAB separates colour from brightness much better than BGR.
    # Blur first so single noisy pixels do not change the median.
    lab = cv2.cvtColor(cv2.GaussianBlur(work, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    pm = None
    if "belt_lab" in roi:                           # moving camera: find the belt again each frame
        pm = _belt_region(lab, roi["belt_lab"])
    elif "poly" in roi:                               # belt polygon: ignore everything outside it
        pm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(pm, [np.array(roi["poly"], np.int32) - np.array([ox, oy])], 255)
    belt = np.median(lab[pm > 0] if pm is not None else lab.reshape(-1, 3), axis=0)   # belt colour

    # How far is every pixel from that belt colour? norm() gives the length of the vector.
    diff = np.linalg.norm(lab - belt, axis=2)
    if pm is not None:
        diff[pm == 0] = 0                           # outside the belt counts as "belt"
    d8 = np.clip(diff * 4, 0, 255).astype(np.uint8)   # scale 0..63 up to 0..255

    # --- step 2: split into "belt" and "not belt" ---
    if plate is not None:
        if plate.shape[:2] != frame.shape[:2]:
            plate = cv2.resize(plate, (frame.shape[1], frame.shape[0]))
        mask = scene_check.plate_mask(work, plate[oy:oy + h, ox:ox + w])
    elif "hsv_lo" in roi:
        mask = _cardboard_mask(work, roi, ox, oy)   # cardboard colour inside the belt polygon
    elif "belt_lab" in roi:
        # Belt pixels = belt hue AND mid brightness (so black and white parcels are NOT belt).
        ab = np.linalg.norm(lab[:, :, 1:] - np.array(roi["belt_lab"][1:], np.float32), axis=2)
        beltpix = (ab < 6.0) & (lab[:, :, 0] > 45) & (lab[:, :, 0] < 150)
        mask = ((~beltpix) & (pm > 0)).astype(np.uint8) * 255
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))   # drop 1px belt-edge rims
        hs = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)          # workers' hi-vis vests are never boxes
        mask[(hs[:, :, 0] >= 22) & (hs[:, :, 0] <= 45) & (hs[:, :, 1] > 110) & (hs[:, :, 2] > 140)] = 0
    else:
        t = otsu_threshold(d8)
        t = max(t, MIN_COLOR_DIFF * 4)               # never segment pure noise
        mask = (d8 > t).astype(np.uint8) * 255

    # --- step 3: clean the mask ---
    k = max(3, int(round(min(w, h) * 0.012)) | 1)   # kernel size from the frame size
    mask = clean_mask(mask, k)
    mask = fill_holes(mask)                          # stickers / barcodes must not count

    # --- step 4: one blob at a time ---
    min_area = min_area_frac * w * h
    max_area = max_area_frac * w * h
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []

    gray_w = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray_w, (3, 3), 0), 40, 110)
    hsv_w = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    for i in range(1, n):                 # component 0 is the background, skip it
        x, y, bw, bh, area = stats[i]
        if area < min_area or area > max_area:
            continue                      # too small = noise, too big = the whole belt

        # Cut this one blob out of the label image so the work stays small and fast.
        comp = ((cc[y:y + bh, x:x + bw] == i).astype(np.uint8)) * 255

        # A blob that is not box-shaped is cut (corner notch / straight edge / watershed);
        # a blob that already fits one quad is left alone (shape_fit.split_blob).
        for reg in shape_fit.split_blob(comp, work[y:y + bh, x:x + bw], int(min_area)):
            c = shape_fit._biggest(reg)
            if c is None:
                continue
            ca = cv2.contourArea(c)
            if ca < min_area:
                REJECT['small'] = REJECT.get('small', 0) + 1
                continue

            fit, sol, quad = shape_fit.fit_score(c)          # B1/B5: is it box shaped?
            if fit < MIN_FIT or sol < MIN_SOLIDITY or not shape_fit.quad_ok(quad):
                REJECT['shape'] = REJECT.get('shape', 0) + 1
                continue
            L, S = shape_fit.quad_dims(quad)
            if L / max(S, 1) > 6:
                REJECT['strip'] = REJECT.get('strip', 0) + 1
                continue                                     # long thin strip, not a box

            reg_m = np.zeros((h, w), np.uint8)
            reg_m[y:y + bh, x:x + bw] = reg
            cf = _context_frac(hsv_w, reg_m)                 # workers / vests / bins
            if cf["hivis"] > MAX_HIVIS_FRAC or (cf["blue"] > MAX_BLUE_FRAC and fit < 0.9):
                REJECT['context'] = REJECT.get('context', 0) + 1
                continue
            es = shape_fit.edge_support(c + np.array([x, y]), edges)
            if es < MIN_EDGE_SUPPORT:
                REJECT['edge'] = REJECT.get('edge', 0) + 1
                continue                                     # no real outline = shadow / blur

            tq = shape_fit.top_face(reg, work[y:y + bh, x:x + bw])   # flat top face, or None
            bx, by, bbw, bbh = cv2.boundingRect(c)  # plain axis-aligned box, crop coords
            gx, gy = bx + x + ox, by + y + oy        # move it back to full-frame coords
            off = np.array([x + ox, y + oy])
            pts = np.round(quad).astype(int) + off                  # quad corners, full frame
            rect = cv2.minAreaRect(c)

            # Cut off by the left or right side of the belt? Then the size is wrong.
            touches = (bx + x) <= 2 or (bx + x + bbw) >= w - 2
            boxes.append({"x": gx, "y": gy, "w": bbw, "h": bbh,
                          "cx": gx + bbw / 2.0, "cy": gy + bbh / 2.0,
                          "rw": float(L), "rh": float(S),            # from the QUAD edges
                          "rw_rect": float(max(rect[1])), "rh_rect": float(min(rect[1])),
                          "pts": pts,
                          "top_quad": (np.round(tq).astype(int) + off) if tq is not None else None,
                          "fit": fit, "solidity": sol, "edge_support": es,
                          "score": float(0.5 * fit + 0.3 * es + 0.2 * sol),
                          "touches_border": bool(touches)})

    # Sort left to right, so the numbers do not jump around between frames.
    boxes.sort(key=lambda b: b["cx"])

    if static is not None:
        boxes, _ = static.filter(frame, boxes)   # the room behind the belt is not a box

    for b in boxes:
        col = (0, 165, 255) if b["touches_border"] else (0, 255, 0)
        cv2.polylines(vis, [b["pts"]], True, col, 2)
    return boxes, vis


def main():
    """Draw the detected boxes on one single frame and save it as a picture."""
    p = argparse.ArgumentParser(description="Preview box detection on one frame.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    p.add_argument("--frame", type=int, default=150)
    p.add_argument("--save", default=os.path.join("assets", "detect_preview.png"))
    p.add_argument("--warmup", type=int, default=60,
                   help="frames run before the preview, so the background filter knows the room")
    p.add_argument("--no-static", action="store_true",
                   help="keep detections that match the still background")
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    v = os.path.join(ROOT, a.video)
    roi = belt_roi.get_roi(v)
    cap = cv2.VideoCapture(v)
    static = None if a.no_static else BackgroundFilter()
    if static is not None:
        for _ in range(max(a.warmup, 0)):            # learn the room first
            ok, f = cap.read()
            if not ok:
                break
            segment_boxes(f, roi, static=static)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)   # jump straight to the frame we want
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("[!] Could not read frame.")
    boxes, vis = segment_boxes(frame, roi, static=static)
    out = os.path.join(ROOT, a.save)
    cv2.imwrite(out, vis)
    print(f"[+] impl={impl.get_mode()}  {len(boxes)} boxes -> {out}   (ROI {roi})")
    if static is not None:
        print(f"    background filter threw away {static.dropped} still "
              f"wall/floor detections during warm-up")
    for b in boxes:
        print(f"    x={b['x']} y={b['y']} w={b['w']} h={b['h']}  "
              f"rotated {b['rw']:.0f}x{b['rh']:.0f}  border={b['touches_border']}")


if __name__ == "__main__":
    main()
