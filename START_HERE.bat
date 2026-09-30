@echo off
setlocal
cd /d "%~dp0"
set LOKY_MAX_CPU_COUNT=4
set V=assets\video\sd_conveyor_2.mp4
title Conveyor Project

echo ==========================================================
echo   CONVEYOR INSPECTION PROJECT - type a number + ENTER
echo   video = %V%
echo ==========================================================
echo.
echo   1  Check everything works        (about 1 minute)
echo   2  Run the full pipeline         (about 1 minute)  ^<- main demo
echo   3  Run in centimetres            (about 1 minute)
echo   4  Run the ALTERNATIVE version   (about 1 minute)
echo   5  Open the output folder
echo   6  Live window on the video
echo   0  Quit
echo.
set "C="
set /p "C=Number: "

if "%C%"=="1" goto s1
if "%C%"=="2" goto s2
if "%C%"=="3" goto s3
if "%C%"=="4" goto s4
if "%C%"=="5" goto s5
if "%C%"=="6" goto s6
if "%C%"=="0" goto fin
echo.
echo   Please type 1 2 3 4 5 6 or 0
pause
goto top

:top
echo.
echo ==========================================================
echo   1 check   2 pipeline   3 cm   4 alt   5 folder   6 live   0 quit
echo ==========================================================
set "C="
set /p "C=Number: "
if "%C%"=="1" goto s1
if "%C%"=="2" goto s2
if "%C%"=="3" goto s3
if "%C%"=="4" goto s4
if "%C%"=="5" goto s5
if "%C%"=="6" goto s6
if "%C%"=="0" goto fin
goto top

:s1
echo.
echo [1] Checking all parts. Please wait.
python box_detect.py --video %V% --frame 229
echo.
python moduleA\segmentation.py
echo.
echo [1] Done. Pictures are now in the assets folder.
pause
goto top

:s2
echo.
echo [2] Full pipeline. Please wait about one minute.
echo.
python pipeline.py --video %V%
echo.
echo [2] Done. The table above is the result.
echo     An annotated video was saved as assets\pipeline_output.mp4
pause
goto top

:s3
echo.
echo [3] Pipeline in centimetres.
echo     Type the camera height in cm above the belt.
echo     If you do not know it, type 120 and press ENTER.
echo.
set "H="
set /p "H=Camera height in cm: "
echo.
python pipeline.py --video %V% --calib moduleB\calibration.json --cam-height-cm %H%
echo.
echo [3] Done.
pause
goto top

:s4
echo.
echo [4] ALTERNATIVE version. Please wait about one minute.
echo.
python pipeline.py --video %V% --impl alt
echo.
echo [4] Done. Nearly the same answer, different maths inside.
pause
goto top

:s5
echo.
echo [5] Opening the assets folder.
explorer.exe "%CD%\assets"
pause
goto top

:s6
echo.
echo [6] Live window. Press q to quit, space to pause, s for a snapshot.
echo.
python live_demo.py --video %V%
echo.
echo [6] Finished.
pause
goto top

:fin
echo.
echo Bye.
endlocal
