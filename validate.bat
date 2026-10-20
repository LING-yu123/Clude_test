@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
rem usage: validate.bat [slope deg] [slope dir] [groups, 0=all] [drive 1/0] [acc time s] [speed scale] [on or off suspension]
"%PY%" tools\validate_plan3.py %*
pause
