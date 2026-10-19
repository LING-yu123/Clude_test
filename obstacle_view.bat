@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
rem usage: obstacle_view.bat [step or gap] [mm] [wheel torque Nm] [on or off] [payload kg]
"%PY%" tools\obstacle_view.py %*
pause
