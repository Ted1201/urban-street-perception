@echo off
chcp 65001 >nul
cd /d "%~dp0"
python prepare_web.py
if errorlevel 1 (pause & exit /b 1)
start "" "%~dp0web-prototype\dist\macau-streets.html"
