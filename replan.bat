@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
rem re-plan group unloading (slow, uses 6 processes) then smooth into continuous paths
"%PY%" core\plan_groups.py
if errorlevel 1 goto :eof
"%PY%" core\smooth_plan.py
pause
