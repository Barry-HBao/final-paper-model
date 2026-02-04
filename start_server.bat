@echo off
REM Start FastAPI server (Windows)
REM Set FORCE_CPU=1 to force CPU mode (useful if CUDA unavailable)
REM Usage: start_server.bat

setlocal
if not defined FORCE_CPU (
  set FORCE_CPU=0
)

REM Optional: activate venv manually before running this script
REM call .\.venv\Scripts\activate

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
endlocal
