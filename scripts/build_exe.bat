@echo off
REM Build dist\plotter_gui.exe via PyInstaller spec.
REM Run `uv sync --extra build` first to install pyinstaller.

uv run -- pyinstaller --clean --noconfirm plotter_gui.spec
if errorlevel 1 goto :err

echo.
echo Build complete: dist\plotter_gui.exe
pause
exit /b 0

:err
echo.
echo BUILD FAILED. See error above.
pause
exit /b 1
