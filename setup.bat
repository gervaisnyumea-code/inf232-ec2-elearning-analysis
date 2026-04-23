@echo off
REM Setup script for Windows (batch). Run as Administrator to allow package installs.

where python >nul 2>nul
if %errorlevel% neq 0 (
  echo Python not found. Attempting to use winget or choco...
  where winget >nul 2>nul
  if %errorlevel%==0 (
    winget install --silent --accept-package-agreements --accept-source-agreements Python.Python.3
  ) else (
    where choco >nul 2>nul
    if %errorlevel%==0 (
      choco install python -y
    ) else (
      echo Please install Python 3.8+ manually and re-run this script.
      pause
      exit /b 1
    )
  )
)

if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
)

call venv\Scripts\activate.bat
pip install --upgrade pip
if exist requirements.txt (
  pip install -r requirements.txt
)

if not exist .env if exist .env.example (
  copy .env.example .env
)

mkdir logs 2>nul
mkdir reports 2>nul
mkdir data\stream 2>nul
mkdir data\models 2>nul
mkdir app\static\icons 2>nul

REM Start streamlit and open browser
start "" cmd /c "python -m streamlit run app/main.py --server.port 8501"
start http://localhost:8501
