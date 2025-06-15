@echo off
echo ===============================
echo Setting up Python environment...
echo ===============================

REM Step 1: Create virtual environment
python -m venv venv

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo Error: Failed to create virtual environment.
    pause
    exit /b
)

REM Step 2: Activate the virtual environment
call venv\Scripts\activate

REM Step 3: Install requirements
echo Installing packages from requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo Error occurred while installing packages.
    pause
    exit /b
)

echo ===============================
echo Setup complete.
echo ===============================
pause
