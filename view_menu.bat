@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
:menu
echo.
echo ===== MuJoCo views =====
echo  1  Unloading, flat ground            (U = start, 1-7 + arrows = jog arm)
echo  2  Unloading on 10 deg slope, side   (slope direction 90)
echo  3  Unloading on 10 deg slope, uphill (slope direction 180)
echo  4  Climb 10 deg slope                (B = brake / drive)
echo  5  Top speed 0.9 m/s on flat ground
echo  6  Step 20 mm
echo  7  Gap 50 mm
echo  8  Gap 90 mm
echo  0  Exit
choice /c 123456780 /n /m "Select: "
if errorlevel 9 goto :eof
if errorlevel 8 "%PY%" tools\obstacle_view.py gap 90 200 on 50 0.1 & goto menu
if errorlevel 7 "%PY%" tools\obstacle_view.py gap 50 200 on 50 0.1 & goto menu
if errorlevel 6 "%PY%" tools\obstacle_view.py step 20 200 on 50 0.1 & goto menu
if errorlevel 5 "%PY%" tools\slope_view.py 0 0.9 on 50 0.6 & goto menu
if errorlevel 4 "%PY%" tools\slope_view.py 10 0.3 on 50 0.6 & goto menu
if errorlevel 3 "%PY%" core\tip_view.py stow 0 off 180 10 & goto menu
if errorlevel 2 "%PY%" core\tip_view.py stow 0 off 90 10 & goto menu
if errorlevel 1 "%PY%" core\tip_view.py stow 0 off 0 0 & goto menu
