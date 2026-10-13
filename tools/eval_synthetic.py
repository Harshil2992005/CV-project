"""Numbers, not opinions: detect + measure the synthetic fixed-camera video and compare with truth."""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.getcwd())
import box_detect, scene_check, size_model

V = sys.argv[1] if len(sys.argv) > 1 else "assets/video/synthetic_fixed.mp4"
gt = json.load(open(os.path.splitext(V)[0] + ".gt.json"))
K = np.array(gt["K"])
plate = scene_check.build_plate(V, force=True)
roi = {"x": 0, "y": 0, "w": 1280, "h": 720}
cap = cv2.VideoCapture(V)
true = {int(k): v for k, v in gt["boxes"].items()}


def gt_id(det, fr):
    """which true box is this detection? nearest true top-face centre."""
    c = np.array(det["top_quad"], float).mean(0) if det.get("top_quad") is not None else np.array([det["cx"], det["cy"]])
    best, bid = 1e9, None
    for g in fr["boxes"]:
        d = np.linalg.norm(np.array(g["top_quad"]).mean(0) - c)
        if d < best:
            best, bid = d, g["id"]
    return bid if best < 60 else None


rows = []
for i in range(0, len(gt["frames"]), 3):
    cap.set(1, i); ok, f = cap.read()
    boxes, _ = box_detect.segment_boxes(f, roi, plate=plate, min_area_frac=0.0015)
    for b in boxes:
        if b["touches_border"] or b["top_quad"] is None:
            continue
        tid = gt_id(b, gt["frames"][i])
        if tid is not None:
            rows.append((i, tid, b))
print("usable detections:", len(rows), " per box:", {k: sum(1 for r in rows if r[1] == k) for k in true})

# reference = box 1, the most central frame
ref_rows = [r for r in rows if r[1] == 1]
if not ref_rows:
    raise SystemExit("no reference box detected")
ref = min(ref_rows, key=lambda r: abs(r[2]["cx"] - 640))[2]
R = true[1]
plane = size_model.PlaneScale.from_reference(ref["top_quad"], R["L"], R["D"], ref_height_cm=R["H"], cam_height_cm=gt["cam_height_cm"])
refs = [r[2] for r in ref_rows]
cam = size_model.Camera3D.from_reference_frames(K, [b["top_quad"] for b in refs], R["L"], R["D"], R["H"])
ppc = ref["rw_rect"] / R["L"]       # the OLD way: one px_per_cm from the silhouette of the reference
print(f"reference box 1: top-face quad -> camera centre {cam.C.round(1)} (true cam (0,-105,85))\n")
est = {k: {"old": size_model.SizeEstimator(), "plane": size_model.SizeEstimator(),
           "plane_h": size_model.SizeEstimator(), "cam3d": size_model.SizeEstimator()} for k in true}
for i, tid, b in rows:
    est[tid]["old"].add(b["rw_rect"] / ppc, b["rh_rect"] / ppc)
    L, S, _ = plane.measure(b["top_quad"]); est[tid]["plane"].add(L, S)
    L, S, _ = plane.measure(b["top_quad"], true[tid]["H"]); est[tid]["plane_h"].add(L, S)
    m = cam.measure(b["top_quad"], b["pts"])
    if m: est[tid]["cam3d"].add(*m)
print(f"{'box':>3} {'true LxD(H)':>14} | {'old 1xscale':>12} {'plane':>12} {'plane+h(given)':>15} {'3D auto':>18}  (cm; median over frames)")
tot = {k: [] for k in ("old", "plane", "plane_h", "cam3d")}
for k, t in true.items():
    L, D = max(t["L"], t["D"]), min(t["L"], t["D"])
    out = []
    for m in ("old", "plane", "plane_h", "cam3d"):
        r = est[k][m].result(3)
        if r is None:
            out.append("n/a"); continue
        e = 100 * (abs(r["L"] - L) / L + abs(r["S"] - D) / D) / 2
        tot[m].append(e)
        out.append(f"{r['L']:.1f}x{r['S']:.1f}" + (f" h{r['H']:.0f}" if m == "cam3d" and r['H'] else "") + f" ({e:.0f}%)")
    print(f"{k:>3} {L:>5.0f}x{D:<3.0f}({t['H']:>2}) | " + " ".join(f"{o:>16}" for o in out))
print("\nmean size error %:", {m: round(float(np.mean(v)), 1) if v else None for m, v in tot.items()})
