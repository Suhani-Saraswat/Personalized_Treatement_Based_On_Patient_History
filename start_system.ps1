# =========================================================================
# Starting MediPersonal AI Platform
# ABESEC Ghaziabad (Session 2023-24)
# Sneha, Tanisha, Suhani Saraswat | Guide: Dr. Anshul Tickoo
# =========================================================================

Write-Host "Starting MediPersonal AI Clinical Decision Support Platform..." -ForegroundColor Cyan

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment and installing dependencies..." -ForegroundColor Yellow
    python -m venv venv
    & ".\venv\Scripts\Activate.ps1"
    pip install -r requirements.txt
}

Write-Host "Launching FastAPI ML Service (XGBoost 3.4.1 + Scikit-Learn)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '.\venv\Scripts\Activate.ps1'; uvicorn ml_service.main:app --host 127.0.0.1 --port 8000 --reload"

Start-Sleep -Seconds 2
Write-Host "Opening MediPersonal AI Web Dashboard..." -ForegroundColor Green
Start-Process "index.html"

Write-Host "MediPersonal AI is fully active!" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host " AUTHENTICATION PORTAL CREDENTIALS:" -ForegroundColor Yellow
Write-Host " -------------------------------------------------------------------------" -ForegroundColor Yellow
Write-Host " 1. Doctor Portal (Dr. Demo - Attending Physician / Admin):" -ForegroundColor White
Write-Host "    Username: drdemo" -ForegroundColor Green
Write-Host "    Password: demo123" -ForegroundColor Green
Write-Host "    (Full clinical CDSS access, What-If simulator, cohort analytics)" -ForegroundColor Gray
Write-Host ""
Write-Host " 2. Patient Portal (Personal Health Records):" -ForegroundColor White
Write-Host "    Patient ID: P001 (or P002, P003, P004, P005)" -ForegroundColor Green
Write-Host "    Password:   patient123" -ForegroundColor Green
Write-Host "    (Scoped strictly to personal data: View, Add, and Delete only)" -ForegroundColor Gray
Write-Host "=========================================================================" -ForegroundColor Yellow
