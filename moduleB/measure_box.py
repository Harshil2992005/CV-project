"""
Module B part 3 of 3 - real size of an UNKNOWN box, using ONE reference box,
under four different camera models.

    python moduleB/measure_box.py moduleB/calibration.json \
        --ref-cm 20 15 --ref-px 200 150 --unk-px 260 180 \
        --cam-height-cm 100 --ref-top-cm 10 --unk-top-cm 20
(leave the numbers out and it asks you for them)

The one idea you need
    Z = camera height MINUS the height of the box's TOP face.
    Z is how far the face we are measuring actually is from the camera. The bigger the
    face looks, the closer it is, so the size in cm = size in px * Z / f.

The four models
    1. Orthographic     one fixed scale taken from the reference box, depth ignored
    2. Weak perspective one fixed scale f / Z0, with Z0 the average depth of the two boxes
    3. Affine           a different scale for x and for y, both taken from the reference
                        (this one can handle non-square pixels)
    4. Full perspective  every box uses its own depth:  size = px * Z / f

Which one is right?
    Full perspective. A taller box has its top face CLOSER to the camera, so it looks
    bigger, and the other three models assume one common scale and over-estimate it. The
    error grows roughly as (depth difference / depth): a ceiling camera at 1 m with a
    10 cm height difference gives about 10 % error. If both boxes are the same height,
    all four give the same answer anyway.
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ask(v, label, n=2):
    """Use the value if we have one, otherwise ask the user for it."""
    if v is not None:
        return v
    return [float(x) for x in input(f"{label} ({n} numbers): ").replace(",", " ").split()]


def main():
    p = argparse.ArgumentParser(description="Real box size from one known reference box.")
    p.add_argument("calib", nargs="?", default=os.path.join("moduleB", "calibration.json"))
    p.add_argument("--ref-cm", type=float, nargs=2, metavar=("W", "H"),
                   help="real width and height of the reference box, in cm")
    p.add_argument("--ref-px", type=float, nargs=2, metavar=("W", "H"),
                   help="pixel width and height of the reference box")
    p.add_argument("--unk-px", type=float, nargs=2, metavar=("W", "H"),
                   help="pixel width and height of the UNKNOWN box")
    p.add_argument("--cam-height-cm", type=float, help="camera height above the belt, in cm")
    p.add_argument("--ref-top-cm", type=float, default=0.0,
                   help="height of the reference box (its top face)")
    p.add_argument("--unk-top-cm", type=float, default=0.0,
                   help="height of the unknown box (its top face)")
    p.add_argument("--video-width", type=float, default=0,
                   help="use this if the video is not as wide as the calibration photos")
    a = p.parse_args()

    with open(a.calib) as f:
        c = json.load(f)
    K = np.array(c["K"])
    # If the video is narrower than the calibration photos, focal length scales too.
    k = (a.video_width / c["image_size"]["width"]) if a.video_width else 1.0
    fx, fy = K[0, 0] * k, K[1, 1] * k
    print(f"f = ({fx:.1f}, {fy:.1f}) px   (calibration RMS {c['rms']:.3f} px)")

    ref_cm = ask(a.ref_cm, "Reference box real W H in cm")
    ref_px = ask(a.ref_px, "Reference box W H in pixels")
    unk_px = ask(a.unk_px, "Unknown box W H in pixels")
    Hc = a.cam_height_cm or ask(None, "Camera height above belt in cm", 1)[0]
    Zr, Zu = Hc - a.ref_top_cm, Hc - a.unk_top_cm        # depth of each TOP face
    if Zr <= 0 or Zu <= 0:
        raise SystemExit("[!] Depth came out as 0 or negative, check --cam-height-cm "
                         "and the box heights.")

    # px per cm, taken from the reference box. sx along the width, sy along the height.
    sx, sy = ref_px[0] / ref_cm[0], ref_px[1] / ref_cm[1]
    Z0 = (Zr + Zu) / 2                                   # average depth of the two boxes
    res = {
        "1 Orthographic":     (unk_px[0] / sx,        unk_px[1] / sx),
        "2 Weak perspective": (unk_px[0] * Z0 / fx,   unk_px[1] * Z0 / fy),
        "3 Affine":           (unk_px[0] / sx,        unk_px[1] / sy),
        "4 Full perspective": (unk_px[0] * Zu / fx,   unk_px[1] * Zu / fy),
    }
    # Sanity check: what does each model say about the REFERENCE box, whose answer
    # we already know? A good model must get this one right.
    chk = {
        "1 Orthographic":     (ref_px[0] / sx, ref_px[1] / sx),
        "2 Weak perspective": (ref_px[0] * Z0 / fx, ref_px[1] * Z0 / fy),
        "3 Affine":           (ref_px[0] / sx, ref_px[1] / sy),
        "4 Full perspective": (ref_px[0] * Zr / fx, ref_px[1] * Zr / fy),
    }
    print(f"\nDepths: reference Z={Zr:.1f} cm, unknown Z={Zu:.1f} cm")
    print(f"\n{'model':<20}{'unknown W x H (cm)':>22}"
          f"{'reference check (true ' + f'{ref_cm[0]:.1f}x{ref_cm[1]:.1f})':>36}")
    for m in res:
        print(f"{m:<20}{res[m][0]:>12.2f} x {res[m][1]:<7.2f}"
              f"{chk[m][0]:>14.2f} x {chk[m][1]:<7.2f}")


if __name__ == "__main__":
    main()
