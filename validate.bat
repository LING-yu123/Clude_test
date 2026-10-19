@echo off
cd /d "%~dp0"
set PY=D:\Users\27662\AppData\Local\Programs\Python\Python311\python.exe
rem usage: validate.bat [slope deg] [slope dir] [boxes] [drive 1/0]  (headless dynamics check)
"%PY%" tools\validate_plan.py %*
pause
