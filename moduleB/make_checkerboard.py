"""
make_checkerboard.py - print a checkerboard that can be used for calibration.

    python moduleB/make_checkerboard.py --cols 9 --rows 6 --square-mm 24 \
           --out assets/checkerboard/board.png

How to use it afterwards
    1. Print at 100 % scale (a "fit to page" print makes the square size wrong).
    2. Stick it on a flat, stiff board.
    3. Measure ONE square with a ruler and put that number into --square-mm.
    4. Take 10-15 photos of it from different angles, all at the SAME resolution.
    5. Feed that folder to calibrate.py.

Note: calibrate.py counts SQUARES, not inner corners. A 9x6 square board has
8x5 inner corners, so use --cols 9 --rows 6 there as well.
"""
import argparse
import os

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Draw a printable checkerboard.")
    p.add_argument("--cols", type=int, default=9, help="number of squares across")
    p.add_argument("--rows", type=int, default=6, help="number of squares down")
    p.add_argument("--square-mm", type=float, default=24.0, help="real side of one square, in mm")
    p.add_argument("--dpi", type=float, default=300, help="printer resolution")
    p.add_argument("--out", default=os.path.join("assets", "checkerboard", "board.png"))
    a = p.parse_args()

    # How many printer PIXELS one millimetre of paper is.
    sq = int(round(a.square_mm / 25.4 * a.dpi))
    margin = sq                                        # white margin all round, so the
    img = np.full((a.rows * sq + 2 * margin,              # outermost squares are visible
                   a.cols * sq + 2 * margin), 255, np.uint8)   # white paper

    for r in range(a.rows):
        for c in range(a.cols):
            if (r + c) % 2 == 0:                      # classic checkerboard colouring
                y, x = margin + r * sq, margin + c * sq
                img[y:y + sq, x:x + sq] = 0             # paint this square black

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, img)
    print(f"[+] saved {a.out}  ({a.cols}x{a.rows} squares, {a.square_mm} mm each at {a.dpi} dpi)")


if __name__ == "__main__":
    main()
