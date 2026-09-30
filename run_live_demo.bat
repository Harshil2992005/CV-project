@echo off
REM ---------------------------------------------------------------------
REM Watch the pipeline LIVE.   q = quit, space = pause, s = snapshot.
REM
REM   run_live_demo.bat            -> base mode (the original code path)
REM   run_live_demo.bat alt        -> alt  mode (the alternative functions)
REM ---------------------------------------------------------------------
cd /d "%~dp0"

if "%1"=="alt" (
    python live_demo.py --video assets/video/sd_conveyor_2.mp4 --impl alt
) else (
    python live_demo.py --video assets/video/sd_conveyor_2.mp4 --impl base
)

pause
