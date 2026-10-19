@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
rem usage: view_susp.bat [speed] [pose] [on or off] [bump height mm]
"%PY%" tools\view_sim.py %*
pause
