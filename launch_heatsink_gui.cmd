@echo off
setlocal
set "HEATSINK_GUI_ENV=C:\Users\laptop\anaconda3\envs\heatsink-gui"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
set "PYTHONHOME="
"%HEATSINK_GUI_ENV%\python.exe" "%~dp0simulation_tool.py"
if errorlevel 1 (
  echo.
  echo Project 1 GUI stopped with an error. The interpreter was:
  echo %HEATSINK_GUI_ENV%\python.exe
  pause
)
endlocal
