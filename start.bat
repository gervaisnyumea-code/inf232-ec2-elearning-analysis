@echo off
REM Start script for Windows (batch)
set PORT=8501
if exist .env (
  for /f "usebackq tokens=1* delims==" %%a in (.env) do (
    set "%%a=%%b"
  )
)

if exist venv\Scripts\python.exe (
  start "" venv\Scripts\python.exe -m streamlit run app/main.py --server.port %PORT%
) else (
  start "" python -m streamlit run app/main.py --server.port %PORT%
)
start http://localhost:%PORT%
