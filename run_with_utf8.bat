@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONLEGACYWINDOWSSTDIO=utf-8

echo Running OpenAvatarChat with UTF-8 encoding...
echo Environment variables set:
echo   PYTHONIOENCODING=utf-8
echo   PYTHONUTF8=1
echo   PYTHONLEGACYWINDOWSSTDIO=utf-8
echo.

python run_with_utf8.py %*

pause
