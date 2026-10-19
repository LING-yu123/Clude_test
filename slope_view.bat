@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
rem usage: slope_view.bat [slope deg] [speed m/s] [on or off] [payload kg] [friction]
"%PY%" tools\slope_view.py %*
pause
