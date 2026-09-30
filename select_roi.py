"""
select_roi.py - pick the belt area by dragging a rectangle with the mouse.

Drag a rectangle, then ENTER = accept, R = redo, ESC = quit.
    python select_roi.py --video assets/video/sd_conveyor_2.mp4

The chosen rectangle is saved in assets/belt_roi.json together with the video's name,
so belt_roi.get_roi() uses it automatically from then on.
"""
import argparse
import os

import cv2

import belt_roi
import impl

ROOT = os.path.dirname(os.path.abspath(__file__))


def select_roi(video_path):
    """Show one frame, let the user drag a box, return the ROI (or None)."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("[!] Could not read first frame.")
    H, W = frame.shape[:2]
    s = min(1.0, 1200.0 / W)                       # display scale, so it fits the screen
    down = cv2.resize(frame, None, fx=s, fy=s)
    win = "Drag a box around the belt. ENTER=ok  R=redo  ESC=quit"
    st = {"p1": None, "p2": None, "draw": False}   # the drag state, kept in a dict

    def on_mouse(ev, x, y, flags, param):
        """Mouse callback. Mouse coordinates arrive in the SCALED picture, so we
        keep them as they are and convert them back at the end."""
        if ev == cv2.EVENT_LBUTTONDOWN:            # press -> start a new drag
            st.update(p1=(x, y), p2=(x, y), draw=True)
        elif ev == cv2.EVENT_MOUSEMOVE and st["draw"]:
            st["p2"] = (x, y)                      # move -> update the corner
        elif ev == cv2.EVENT_LBUTTONUP:            # release -> stop drawing
            st["p2"] = (x, y)
            st["draw"] = False

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = down.copy()
        if st["p1"] and st["p2"]:
            cv2.rectangle(vis, st["p1"], st["p2"], (0, 255, 0), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == 13:                                # ENTER
            break
        if k == ord("r"):
            st.update(p1=None, p2=None, draw=False)    # R = forget and start again
        if k == 27:                                # ESC
            cv2.destroyAllWindows()
            return None
    cv2.destroyAllWindows()

    if not st["p1"] or not st["p2"]:
        print("[!] Nothing selected.")
        return None
    (x1, y1), (x2, y2) = st["p1"], st["p2"]
    if abs(x2 - x1) < 20 or abs(y2 - y1) < 20:
        print("[!] Selection too small.")
        return None
    # Convert from screen pixels back to real video pixels (divide by the scale s).
    return {"x": int(min(x1, x2) / s), "y": int(min(y1, y2) / s),
            "w": int(abs(x2 - x1) / s), "h": int(abs(y2 - y1) / s)}


def main():
    p = argparse.ArgumentParser(description="Pick the belt rectangle with the mouse.")
    p.add_argument("--video", default=os.path.join("assets", "video", "sd_conveyor_2.mp4"))
    impl.add_impl_flag(p)
    a = p.parse_args()
    impl.apply_args(a)

    v = os.path.join(ROOT, a.video)
    roi = select_roi(v)
    if roi is None:
        print("[!] No ROI selected - previous one kept.")
        return
    belt_roi.save_roi(roi, v)
    print("Selected belt ROI:", roi)


if __name__ == "__main__":
    main()
