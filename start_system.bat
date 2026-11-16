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
echo =========================================================================
echo  AUTHENTICATION PORTAL CREDENTIALS:
echo  -------------------------------------------------------------------------
echo  1. Doctor Portal (Dr. Demo - Admin):
echo     Username: drdemo
echo     Password: demo123
echo     (Full access to 7 CDSS modules, What-If simulator, all patient cohorts)
echo.
echo  2. Patient Portal (Personal Health Records):
echo     Patient ID: P001 (or P002, P003, P004, P005)
echo     Password:   patient123
echo     (Scoped to personal data: View, Add, and Delete personal records)
echo =========================================================================
pause
