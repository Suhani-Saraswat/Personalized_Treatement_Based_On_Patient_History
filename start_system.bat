@echo off
echo =========================================================================
echo  Starting MediPersonal AI Platform
echo  ABESEC Ghaziabad (Session 2023-24)
echo  Sneha, Tanisha, Suhani Saraswat ^| Guide: Dr. Anshul Tickoo
echo =========================================================================

echo Checking Python virtual environment...
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    call .\venv\Scripts\activate
    pip install -r requirements.txt
)

echo Starting FastAPI ML Engine on http://localhost:8000 ...
start "MediPersonal ML Service" cmd /k ".\venv\Scripts\activate && uvicorn ml_service.main:app --host 127.0.0.1 --port 8000 --reload"

echo Opening Web Application in Browser...
timeout /t 2 >nul
start index.html

echo System is running!
pause
