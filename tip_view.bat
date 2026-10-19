@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
rem usage: tip_view.bat [pose] [payload kg] [on or off] [slope direction] [start slope]
"%PY%" core\tip_view.py %*
pause
