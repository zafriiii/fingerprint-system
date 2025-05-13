@echo off

:: Ensure Python 3.10.11 is installed and on PATH
where python

:: Delete any previous virtual environment
rmdir /s /q venv

:: Create new virtual environment
python -m venv venv

:: Activate it
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install AI libraries
pip install tensorflow==2.15.0
pip install torch torchvision torchaudio

:: Install fingerprint image processing tools
pip install opencv-python scikit-image matplotlib scikit-learn

:: Install cybersecurity and privacy tools
pip install tensorflow-privacy flwr syft cryptography

:: Install fingerprint matching engine
pip install fingerprint-recognition

:: Optional: Notebook environment
pip install notebook

echo Setup complete. Activate anytime with: venv\Scripts\activate
pause
