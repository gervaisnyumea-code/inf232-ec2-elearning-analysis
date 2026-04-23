# PowerShell start script for INF232
Param(
    [int]$Port = 8501
)

# Load .env into environment
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -and $_ -notmatch '^\s*#') {
            $parts = $_ -split '='
            if ($parts.Length -ge 2) {
                $k = $parts[0].Trim(); $v = ($parts[1..($parts.Length-1)] -join '=').Trim('"').Trim("'")
                $env:$k = $v
            }
        }
    }
}

# Start Streamlit using venv python if present
$python = "venv\Scripts\python.exe"
if (Test-Path $python) {
    Start-Process -FilePath $python -ArgumentList "-m streamlit run app/main.py --server.port $Port"
} else {
    Start-Process -FilePath python -ArgumentList "-m streamlit run app/main.py --server.port $Port"
}
Start-Sleep -Seconds 3
Start-Process "http://localhost:$Port"
