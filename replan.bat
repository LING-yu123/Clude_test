@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
rem re-plan the unloading motion (slow, uses all CPU cores)
"%PY%" core\plan_motion.py %*
pause
