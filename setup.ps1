# PowerShell setup script for INF232
# Run as Administrator if installing system packages

Param()

function Exec($cmd) {
    Write-Host "> $cmd"
    iex $cmd
}

# Check Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "Python not found. Attempting to install via winget/choco..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Exec "winget install --silent --accept-package-agreements --accept-source-agreements Python.Python.3"
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Exec "choco install python -y"
    } else {
        Write-Warning "No winget/choco found. Please install Python 3.8+ manually and re-run this script."
        exit 1
    }
}

# Create venv
if (-not (Test-Path venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate and install requirements
$activate = "venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    Write-Host "Activating venv and installing requirements..."
    & $activate; pip install --upgrade pip
    if (Test-Path requirements.txt) { pip install -r requirements.txt }
} else {
    Write-Warning "venv activation script not found. Ensure Python & venv are available."
}

# Copy .env
if (-not (Test-Path .env) -and (Test-Path .env.example)) {
    Copy-Item .env.example .env -Force
    Write-Host "Copied .env.example -> .env"
}

# Create folders
New-Item -ItemType Directory -Force -Path logs, reports, data\stream, data\models, app\static\icons | Out-Null

# Start app and open browser
Write-Host "Starting Streamlit..."
Start-Process -FilePath python -ArgumentList "-m streamlit run app/main.py --server.port 8501"
Start-Sleep -Seconds 3
Start-Process "http://localhost:8501"

Write-Host "Setup complete."
