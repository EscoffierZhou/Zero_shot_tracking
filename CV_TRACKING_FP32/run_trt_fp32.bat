@echo off
echo ========================================================
echo   Launching OSTrack Ferrari (TensorRT FP32 Edition)
echo   Running in WSL (Windows Subsystem for Linux)...
echo ========================================================

wsl /home/escoffierzhou_0523/opencv_build/venv_fix/bin/python "$(wslpath -a .)/app_TensorRT.py"

echo ========================================================
echo   App closed.
echo ========================================================
pause
