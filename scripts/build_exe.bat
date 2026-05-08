@echo off
REM ポータブル exe (dist/plotter_gui.exe) をビルドする。
REM 事前に `uv sync --extra build` で pyinstaller を入れておく。

uv run pyinstaller --clean --noconfirm plotter_gui.spec
echo.
echo Build complete: dist\plotter_gui.exe
pause
