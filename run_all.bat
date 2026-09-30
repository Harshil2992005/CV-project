@echo off
REM ---------------------------------------------------------------------
REM Runs every module of the conveyor project, one after another.
REM
REM   run_all.bat            -> base mode (the original code path)
REM   run_all.bat alt        -> alt  mode (the alternative functions)
REM
REM Press ENTER between steps so you can read the output.
REM ---------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set MODE=%1
if "%MODE%"=="" set MODE=base

echo =============================================================
echo   Conveyor inspection - full run, impl = %MODE%
echo =============================================================

echo.
echo --- 1/9  Module A: five segmentation methods ---------
pause
python moduleA\segmentation.py --impl %MODE%

echo.
echo --- 2/9  Module B1: camera calibration ---------------
pause
python moduleB\calibrate.py assets\reference_set --cols 10 --rows 7 --square-mm 24 --out moduleB\calibration.json --impl %MODE%

echo.
echo --- 3/9  Module B2: P = K [R|t] decomposition --------
pause
python moduleB\decompose_P.py moduleB\calibration.json --impl %MODE%

echo.
echo --- 4/9  Module C: optical flow and belt speed -------
pause
python moduleC\optical_flow.py --impl %MODE%

echo.
echo --- 5/9  Module D: Kalman tracking -------------------
pause
python moduleD\kalman_tracker.py --impl %MODE%

echo.
echo --- 6/9  Module E: box recognition -------------------
pause
python moduleE\recognize.py --impl %MODE%

echo.
echo --- 7/9  Belt ROI ------------------------------------
pause
python belt_roi.py --impl %MODE%

echo.
echo --- 8/9  Shared box detector -------------------------
pause
python box_detect.py --impl %MODE%

echo.
echo --- 9/9  Full pipeline, modules 1 to 5 ---------------
pause
python pipeline.py --impl %MODE%

echo.
echo --- Live demo, headless, 200 frames ------------------
pause
python live_demo.py --headless 200 --impl %MODE%

echo.
echo =============================================================
echo   Done. All images and videos are in the assets folder.
echo =============================================================
endlocal
