@echo off
rem Launch the Orchard control panel.
cd /d "%~dp0"
start "" pythonw orchard_gui.py
if errorlevel 1 python orchard_gui.py
