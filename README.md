# Camera-Based Conveyor Inspection & Tracking System

Classical OpenCV only (no deep learning). One camera watches a conveyor belt; from the video we get
**where** each box is, **how big** it is (cm), **how fast** the belt moves and **what type** each box is.

```
frame -> A segment -> D Kalman tracks -> B px->cm -> C belt speed (optical flow) -> E box type
```

## Quick start

```bash
pip install -r requirements.txt

python belt_roi.py --video assets/video/sd_conveyor_2.mp4 --save   # once per video
python pipeline.py --video assets/video/sd_conveyor_2.mp4          # all 5 modules chained
python live_demo.py                                               # live window
```

On Windows, `START_HERE.bat` or `run_all.bat` runs every module in order and pauses
between steps so the output can be read. Add `alt` as the argument (`run_all.bat alt`)
to run the alternative code paths instead.

## Contents

| Section | What is inside |
|---|---|
| [Folder layout](#folder-layout) | what every file and folder is for |
| [Modules](#modules-run-each-alone) | the five modules, A to E, and the command for each |
| [Module B - camera calibration](#module-b---camera-calibration-the-one-step-that-needs-you-in-the-loop) | the one step that needs your own photos |
| [Full pipeline](#full-pipeline) | A + D + B + C + E chained together |
| [What each module demonstrates](#what-each-module-demonstrates-max-5-sentences-each) | five sentences per module |
| [What is new in v2](#what-is-new-in-v2) | the `--impl base` / `--impl alt` switch |

## Folder layout
```
CV-project/
|-- assets/
|   |-- video/            4 clips; sd_conveyor_2.mp4 is the default
|   |-- checkerboard/     board.html + board.png (printable) + YOUR photos
|   |-- reference_set/    OpenCV's public sample images - reference only
|   |-- crops/0,1,2       box samples per type, for Module E
|   |-- belt_roi.json     belt area, bound to one video
|   `-- *.png / *.mp4     outputs written by the modules
|-- moduleA/  segmentation          5 methods compared on the same frames
|-- moduleB/  make_checkerboard     prints the calibration board
|            calibrate             recovers K + distortion  <- needs your photos
|            measure_box           box size in cm under 4 camera models
|            decompose_P           P = K[R|t], and unpacking it
|-- moduleC/  optical_flow          hand-built Lucas-Kanade + belt speed
|-- moduleD/  kalman_tracker        Kalman filter from scratch
|-- moduleE/  recognize             eigenboxes / alignment / Hu moments
|-- belt_roi.py  box_detect.py  select_roi.py     shared helpers
|-- pipeline.py  live_demo.py  run_all.bat  run_live_demo.bat  all modules chained
|-- impl.py                     the base / alt switch (new in v2)
|-- calibaration.ipynb               Module B, step by step (learning)
`-- Building a ... .docx             the written report
```

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/Harshil2992005/CV-project.git
cd CV-project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

| Dependency | Used for |
|---|---|
| `opencv-python` | camera, `calibrateCamera`, `findHomography`, drawing |
| `numpy` | all the array maths |
| `scipy` | RQ decomposition, Hungarian matching |
| `matplotlib` | the saved plots |
| `scikit-image` | watershed, Felzenszwalb, mean shift, N-Cut, Otsu |
| `scikit-learn` | PCA for eigenboxes |

Put your videos in `assets/video/` (default: `sd_conveyor_2.mp4`, change with `--video`).
Real footage only (own phone video or a public clip).

## Step 0 - belt ROI (do this once per video)
```
python belt_roi.py --video assets/video/sd_conveyor_2.mp4 --save     # automatic
python select_roi.py --video assets/video/sd_conveyor_2.mp4          # or drag it by hand (best)
python box_detect.py --video assets/video/sd_conveyor_2.mp4 --frame 150   # check the detection
```
The ROI file remembers its video, so it is never applied to another video by mistake.
Green = clean box, orange = box cut by the belt edge (its size is NOT measured).

## Belt speed needs `--gap 8`, not the default 1

The belt moves ~0.3 px per frame, which is at the noise floor of sparse
optical flow. With the default `--gap 1` the estimate is off by ~40 %.
`--gap 8` averages the motion over 8 frames and is much more stable:

| `--gap` | reported | measured truth |
|---|---|---|
| 1 | 12.0 px/s | 8.6 px/s |
| 4 | 6.9 px/s | 8.6 px/s |
| 8 | **10.4 px/s** | 8.6 px/s |

(Truth = median corner displacement over 100 frames with a forward-backward
check, i.e. 4 s of video, so the motion is large enough to measure reliably.)

Two more things worth knowing:

* The belt surface in these clips is almost textureless, so very few trackable
  corners sit on it. Most corners in the belt ROI are on the **static
  background**, and RANSAC keeps the largest consensus - which is the static
  part. That is why the reported inlier ratio is low (50/128); it is expected,
  not a bug. `optical_flow.py` fits the model on the largest clean **box**
  (a box rides the belt and is textured), which is why it is the accurate path.
* `pipeline.py` prints its own belt speed from `estimate_belt_motion()`, which
  fits the **whole ROI** instead of a box, so that number reads low. Use the
  standalone `optical_flow.py --gap 8` figure for the report.

## Modules (run each alone)
| Module | Command | Output |
|---|---|---|
| A | `python moduleA/segmentation.py` | `assets/segmentation_grid.png` |
| B | `python moduleB/make_checkerboard.py` -> `python moduleB/calibrate.py assets/checkerboard` -> `python moduleB/measure_box.py moduleB/calibration.json` -> `python moduleB/decompose_P.py moduleB/calibration.json` | `calibration.json`, `assets/undistort_demo.png`, 4-model size table |
| C | `python moduleC/optical_flow.py --px-per-cm 20 --gap 8` | `flow_arrows.png`, `flow_sparse.png`, `flow_affine.png`, belt speed |
| D | `python moduleD/kalman_tracker.py` | `kalman_plot.png`, `kalman_overlay.mp4`, printed predict/measure/update |
| E | `python moduleE/recognize.py` (`--recollect` to redo crops) | crops, `eigenboxes_eigenvectors.png`, accuracy table |

## Module B - camera calibration (the one step that needs you in the loop)

Every other module runs as-is. Module B cannot: it needs **real photos of a
printed checkerboard**, and those only you can shoot. This is the whole checklist.

### Step 1 - print the board
```
python moduleB/make_checkerboard.py --cols 9 --rows 6 --square-mm 24
```
Prints `assets/checkerboard/board.png` (9x6 squares, 24 mm each, 300 dpi).

> **`--cols` and `--rows` are SQUARES, not corners.** A 9x6 grid of squares has
> **8x5 inner corners** - `calibrate.py` does that conversion itself
> (`pattern = (cols-1, rows-1)`). So do **not** "correct" the flags to `--cols 8
> --rows 5` because you read somewhere that a 9x6 board has 8x5 corners. Pass the
> number of **squares** you printed, exactly as shown above.
>
> Verified on this exact file: detected cleanly at pattern (8, 5), 283.00 px per
> square = 24.0 mm, whole sheet 263.6 x 191.7 mm, so it prints on one A4.

* **Print at 100 % scale.** Turn OFF *fit to page* / *shrink to fit*. If the
  printer scales it, every square is the wrong size and K comes out wrong.
* **Measure one square with a ruler.** If it is not 24 mm, re-run with the real
  number: `--square-mm 23.8`, say. This value is the truth you are calibrating
  against, so do not trust the printer dialog.
* Print on **plain matte paper**, not glossy - gloss creates reflections that
  hide corners.

### Step 2 - take 15-20 photos into `assets/checkerboard/`
* **Move the camera, not the board** - or if the camera is on a tripod, rotate
  and tilt the board instead. Either way you need *variety*, not 20 copies of
  the same pose.
* Tilt the board left/right, up/down, and rotate it in-plane.
* Keep the **whole board visible** with a small margin around it.
* **One resolution, one camera.** `calibrate.py` skips any photo whose size
  differs from the first one, so a phone that switches modes between photos will
  silently throw half of them away.
* Sharp, no motion blur, decent even light. Delete any blurry one.
* Vary the distance too - some near, some further.

### Step 3 - learn it interactively (optional but recommended)
Open `calibaration.ipynb` and run the cells top to bottom. It walks through what
an image is, how corners are found, what K and `dist` mean, and it prints the
same K that `calibrate.py` will produce. Useful when a step below fails and you
want to know *which* stage is unhappy.

### Step 4 - calibrate
```
python moduleB/calibrate.py assets/checkerboard --cols 9 --rows 6 --square-mm 24 ^
       --out moduleB/calibration.json
```
It prints one line per photo (`[+] name` = accepted, `[!] skip ...` = rejected
with the reason) and finishes with the RMS re-projection error.

* **Success = RMS < 1.0 px.** Above 1.0, the result is not trustworthy - the usual
  causes are a mis-sized print, too few good photos, blurry shots, or photos all
  from nearly the same angle.
* It refuses to run with fewer than 3 accepted photos (10-15 is the comfortable
  minimum). If it says *"Need at least 3 good photos"*, read the per-photo skip
  lines above - they tell you exactly which ones failed and why.
* It also writes `assets/undistort_demo.png` (before | after side by side).
  **Look at it.** If the "after" image does not have straight, unwarped lines
  where the board's squares are, something is wrong even if RMS looks okay.
* **Do not rename or re-copy `board.png` in there.** It is the *printable* board,
  not a photo. It is flat and perfect, so corners are found in it and it will
  quietly poison the calibration. `calibrate.py` skips the exact name
  `board.png` - so a copy named `my_copied_board.png` is *not* skipped and will
  ruin the result.

### Step 5 - decompose P = K[R|t]
```
python moduleB/decompose_P.py moduleB/calibration.json
```
Builds P = K[R|t] and unpacks it again with an RQ decomposition, then
cross-checks against `cv2.decomposeProjectionMatrix`. Run it with **no argument**
if you have no photos yet - it falls back to a synthetic example that still
demonstrates the maths.

### Step 6 - measure a real box in cm
```
python moduleB/measure_box.py moduleB/calibration.json
```
It asks for four things: the reference box's real size in cm, its size in pixels,
the unknown box's size in pixels, and the camera height above the belt. It then
prints the unknown box under four camera models. **Full perspective (model 4) is
the accurate one** - it divides by that box's own depth, while the other three
assume one shared scale and drift as size or depth moves away from the reference.

### Step 7 - feed the scale back into the pipeline
```
python pipeline.py --calib moduleB/calibration.json --cam-height-cm 100 --box-height-cm 10
```
`--cam-height-cm` = your camera's height above the belt (measure it).
`--box-height-cm` = box height, used to correct for a box's top face being closer
to the camera. Both are needed for the perspective-corrected scale.

Until Step 4 succeeds, everything else runs in **pixels** and says so
(`NOTE: no scale given -> pixels`), which is fine for modules A, C, D and E.

### Reference run - proving the code works before you have photos

So that Module B is runnable and demonstrable today, the repo ships a
**reference calibration** built from OpenCV's own public sample images.

> **This is NOT your camera.** `assets/reference_set/` holds `left01.jpg` -
> `left14.jpg` downloaded from the official `opencv/opencv` repo
> (`samples/data`, 640x480). The saved file is
> `moduleB/calibration_reference.json` - deliberately named `_reference` so it can
> never be mistaken for your own `moduleB/calibration.json`.
>
> Its only purpose is to prove the calibration code is correct. It must **not** be
> used to state real centimetres for your conveyor video - see
> "Absolute cm and third-party footage" below.

Reproduce it:

```
python moduleB/calibrate.py assets/reference_set --cols 10 --rows 7 --square-mm 24 ^
       --out moduleB/calibration_reference.json
```

Result - all 13 images accepted, well under the 1 px target:

```
RMS re-projection error: 0.2404 px   (aim < 1.0)
K = [[532.535, 0, 341.648], [0, 532.571, 232.712], [0, 0, 1]]
dist (k1,k2,p1,p2,k3) = [-0.30795, 0.16384, 0.00077, 0.00046, -0.0454]
```

`decompose_P.py` round-trips it to machine precision and agrees with OpenCV:

```
max|K-K2|=2.27e-13  max|R-R2|=4.44e-16  max|t-t2|=1.71e-13
det(R2) = 1.0  (+1, so a proper rotation)
OpenCV decomposeProjectionMatrix K matches: True
```

**The four size models are only as good as your inputs.** If you claim a 20 cm box
is 160 px wide, that pins the depth, and the perspective models will only
reproduce 20 cm if the height you give matches it:

```
Z = real_cm * f / px = 20 * 532.535 / 160 = 66.57 cm
```

Pass `--cam-height-cm 66.57` and all four models agree exactly, which is the
cheapest self-test the module has:

```
model                   unknown W x H (cm)    reference check (true 20.0x15.0)
1 Orthographic             15.00 x 11.25           20.00 x 15.00
2 Weak perspective         15.00 x 11.25           20.00 x 15.00
3 Affine                   15.00 x 11.25           20.00 x 15.00
4 Full perspective         15.00 x 11.25           20.00 x 15.00
```

Pass an inconsistent height (e.g. 100) and models 2 and 4 report 30.04 cm for a
box that is really 20 cm - the reference check is what exposes it, so always read
that column.

K is resolution-dependent, and `measure_box.py --video-width` rescales it for you.
Same physical setup, referred to a 768 px wide video instead of 640:

```
f = (639.0, 639.1) px      (was 532.5 at 640 px;  768/640 = 1.2)
model 4 Full perspective   15.00 x 11.25 cm      unchanged, as it must be
```

### Absolute cm and third-party footage

`assets/video/sd_conveyor_2.mp4` is third-party stock footage (768x432, 25 fps,
generic `isom` MP4, no EXIF and no camera make/model). Its intrinsics are
therefore **unknown and unrecoverable** - the re-encode means even the original
resolution is gone. Stating a centimetre value for a box in that video would be
an assumption, not a measurement.

So, stated honestly:

* **Modules A, C, D, E** run on the public clip and report **pixels**.
* **Module B** is a real, self-captured calibration on your **own** phone
  (`moduleB/calibration.json`, once you have taken the photos). That one is
  genuine and reportable.
* **Real centimetres** come from `measure_box.py` applied to imagery whose camera
  you calibrated - e.g. one photo of a box you measured with a ruler, shot from a
  height you measured, with `moduleB/calibration.json`. Run it on its own, not
  through `pipeline.py` on the stock clip.

The belt speed in cm/s elsewhere in this README is a **scale assumption**
(`--px-per-cm`), not a calibration - it is a separate, clearly labelled number.

---

## Full pipeline
```
python pipeline.py --video assets/video/sd_conveyor_2.mp4 --px-per-cm 20.5
python pipeline.py --calib moduleB/calibration.json --cam-height-cm 100 --box-height-cm 10
python live_demo.py            # live window (or run_live_demo.bat on Windows)
```
Prints per box: ID, size, speed, type, and saves `assets/pipeline_output.mp4`.
Without `--px-per-cm` (or `--calib` + `--cam-height-cm`) everything is in **pixels** - real cm need a scale.

## What each module demonstrates (max 5 sentences each)
**A.** Five methods run on the same 5 frames: watershed, split & merge (from scratch), Felzenszwalb, mean shift and normalized cut.
Colour-only methods (split & merge, Felzenszwalb, mean shift) merge touching boxes of the same colour into one region.
Watershed uses a distance-transform marker per box, so it separates touching boxes best, and it is what the shared detector uses.
N-Cut is global and heavy, so it runs on a small superpixel graph.

**B.** `calibrateCamera` recovers K (focal length, principal point) and radial distortion from checkerboard photos; success is RMS < 1 px.
`decompose_P.py` builds P = K[R|t] and recovers K, R, t again with an RQ decomposition.
With one reference box of known size, the unknown box is measured under orthographic, weak-perspective, affine and full-perspective models.
Full perspective is the most accurate because it divides by each box's own depth; the other three assume one common scale.

**C.** Lucas-Kanade is solved from scratch (2x2 normal equations per window, coarse-to-fine) and compared with OpenCV sparse corner flow.
On flat box faces dense flow is unreliable (aperture problem), while sparse corners stay robust.
A 6-parameter affine flow model fitted with RANSAC rejects outlier vectors (inliers green, outliers red).
Belt speed = flow at the region centre x fps / (px per cm).

**D.** A Kalman filter (state x, y, vx, vy; constant velocity) blends noisy detections with a motion prediction, giving a smoother path.
The predict -> measure -> update cycle is printed for 13 consecutive frames including the occlusion.
During the 3-5 frame dropout the filter keeps tracking with prediction only, which demonstrates observability.
The tracker matches boxes to tracks with the Hungarian algorithm and only starts a track for unmatched boxes.

**E.** Crops of 3 box types are cut from the video (clustered by size) or supplied by hand in `assets/crops/<type>/`.
Eigenboxes (PCA + nearest neighbour), alignment (ORB + RANSAC homography residual), Hu moments and a size classifier are compared on held-out, rotated, darker and brighter test crops.
Alignment and Hu moments survive lighting changes / rotation better than eigenboxes, which store raw brightness and orientation.
Hu moments are scale invariant, so they separate types only when the types differ in shape, not just in size.

---

## What is new in v2

This repo is a re-write of an earlier private version of the project. Every file has been given
plain-English comments explaining the maths, and about thirty places now contain a
**second way of doing the same thing**. Nothing is required: if you never type
`--impl alt`, you get the original code and the original output.

### The one switch: `--impl`

Every script that contains an alternative takes one extra argument:

```
python <script> --impl base     the original code path      (this is the default)
python <script> --impl alt      the alternative functions
```

`impl.py` holds the whole mechanism. It is only 4 lines of real logic:

```python
MODE = "base"                            # the one line you can edit
CHOICES = {"base", "alt"}

def is_alt():
    return MODE == "alt"

def dispatch(base_fn, alt_fn, *args, **kwargs):
    return alt_fn(*args, **kwargs) if is_alt() else base_fn(*args, **kwargs)
```

`impl.py` is imported by every module, so setting `MODE = "alt"` in that one file
switches the whole project over. `run_all.bat alt` does it from the command line
instead, without editing anything.

Scripts with no alternatives (`moduleB/measure_box.py`, `moduleB/make_checkerboard.py`)
have no `--impl` flag, because there is nothing to switch.

### Run everything

```
run_all.bat           all modules, base mode, pauses between steps
run_all.bat alt       all modules, alt mode
```

Or step by step:

```
python moduleA/segmentation.py
python moduleB/calibrate.py assets/reference_set --cols 10 --rows 7 --square-mm 24 --out moduleB/calibration.json
python moduleB/decompose_P.py moduleB/calibration.json
python moduleB/measure_box.py --ref-cm 20 12 --ref-px 100 60 --unk-px 140 84 --cam-height-cm 120
python moduleC/optical_flow.py
python moduleD/kalman_tracker.py
python moduleE/recognize.py
python belt_roi.py
python box_detect.py
python pipeline.py
python live_demo.py
```

Add `--impl alt` to any of them to see the alternatives.

### The alternatives, and what actually changes

| Where | base (original) | alt (alternative) | Same output? |
|---|---|---|---|
| `box_detect.otsu_threshold` | histogram loop | `skimage.filters.threshold_otsu` | yes, to the last bit |
| `box_detect.open_or_closed` | NumPy binary dilation | `cv2.morphologyEx` | yes |
| `box_detect.fill_holes` | flood fill | convex-hull fill | yes on these boxes |
| `box_detect.rect_from_contour` | `cv2.minAreaRect` | hand-rolled corner search | yes |
| `belt_roi.smooth_mask` | box-blur in NumPy | `cv2.blur`, zero-padded | yes |
| `moduleA` drawing | `skimage.segmentation.mark_boundaries` | NumPy label-neighbour test | yes |
| `moduleA` watershed markers | `peak_local_max` on the distance map | OpenCV good-features + own sub-pixel peak | no, alt finds its own markers |
| `moduleB` corner finding | `cv2.findChessboardCornersSB` | classic `findChessboardCorners` | no, RMS 0.24 px vs 0.41 px |
| `moduleB` RQ | `scipy.linalg.rq` | hand-written from `np.linalg.qr` | yes to 1e-13 |
| `moduleC` gradients | `skimage.color.rgb2gray` + Sobel | `cv2.cvtColor` + `cv2.Sobel` | no, last-digit noise |
| `moduleC` affine fit | NumPy RANSAC | `cv2.findHomography` with RANSAC | no, ~5% different inliers |
| `moduleD` covariance | Joseph form `(I-KH)P(I-KH)T+KRKT` | the short form `(I-KH)P` | no, alt is slightly tighter |
| `moduleD` matching | Hungarian (`linear_sum_assignment`) | greedy cheapest-first | no, see the count below |
| `moduleE` lighting test | multiply the brightness | gamma (`LUT`) | no, different darkening |
| `moduleE` Eigenboxes | `np.linalg.svd` | `sklearn.decomposition.PCA` | yes |
| `moduleE` size classifier | class median | 1-nearest-neighbour | no, 1 of 37 boxes changes type |

Where the table says **no**, the two answers are both reasonable and both run; they
just are not the same number, because the algorithm underneath is different. That is
the point of having them.

#### What the pipeline actually looks like in each mode

This is measured, not guessed. Both modes were run on `sd_conveyor_2.mp4` and the
printed box tables were compared, matching boxes to each other by size:

```
tracks found:            base 37        alt 38
same physical box:       36 of 37 matched one-to-one (within 1-2 px)
differ ONLY in the ID:   19 of 36      <- just renumbering, cosmetic
same frame count:        33 of 36
same size:               34 of 36
same speed:              33 of 36
same TYPE:               35 of 36
```

So of 37 boxes, **34 are reproduced exactly** and only three really move:

| box | base | alt | why |
|---|---|---|---|
| 153x120 | id 66, 24 frames | id 68, 12 frames | greedy matching split this track in half |
| 279x181 | id 73, 18 frames | id 67, 25 frames, 279x183 | greedy matching glued ~7 more frames on |
| 88x31 | id 79, **medium** | id 80, **small** | the 1-NN size classifier, not the tracker |
| 113x81 | id 98, 22 frames | id 95, 26 frames | greedy matching kept it alive 4 frames longer |

`alt` also invents one extra short track, `83x27`, 8 frames long, which has no partner
in base. That single spurious track is why the alt IDs drift upwards (66 becomes 68,
70 becomes 71, and so on) - greedy made one extra decision somewhere around frame 200,
and every later number shifts by one. Belt speed is identical at `1.5 px/s`.

None of this is a bug. Greedy matching and Hungarian matching are both valid answers
to the same question, and they disagree exactly when one cheap pair has to be weighed
against a better overall set of pairs. On a cluttered belt that happens; on clean
footage they agree.


### How base mode was checked

Every module was run in `base` mode and its output compared line by line with the
original project:

* Module A, B, C, D, E printouts: **identical** to the original.
* `pipeline.py`: the tracked-box table is **identical**; only the new
  `[pipeline] impl=base` header line is extra.
* `belt_roi.py`: identical (`{'x': 0, 'y': 0, 'w': 768, 'h': 432}`).
* `box_detect.py`: identical box lists on frames 76, 229, 383, 536, 689.

Two gotchas worth knowing:

* `moduleE/recognize.py --recollect` re-cuts `assets/crops` from the video. The shipped
  crops came from the original project, so re-cutting them shifts every accuracy number
  a little (Eigenboxes goes 58% to 67%, for example). If you only want to look at the
  shipped dataset, do **not** pass `--recollect`.
* `calibaration.ipynb` and the report `.docx` are copied over from the original
  unchanged. They are learning material, not code, and they do not know about `impl.py`.

`assets/belt_roi.json` is **not** shipped. `belt_roi.py` auto-detects the belt from the
video when the file is missing, and on this clip it correctly returns the whole frame.
Run `python belt_roi.py --save` to write the file if you want it fixed to one video.

Three bugs were found and fixed while doing this, all in `alt` code only, so `base`
mode was never at risk:

1. `moduleC` - the OpenCV affine fit was done on raw pixel coordinates, which are
   large numbers and make the 2x2 system badly scaled. The belt speed came out as
   `9601.5 px/s`. It now fits around the point being measured, giving `1.5 px/s`,
   the same as base.
2. `moduleB/decompose_P.py` - `rq_handmade` used the wrong flip order and returned
   garbage (`K` all below 1). The correct identity is `M = J L Q' J` with `J` the
   anti-diagonal; it now matches SciPy to `1e-13`.
3. `moduleA` - the OpenCV border drawing passed an `int32` label image to
   `cv2.Canny`, which only accepts `uint8`, so every method failed in `alt` mode.
   The borders are now found by comparing neighbouring labels, which has no 8-bit
   limit and is faster.
