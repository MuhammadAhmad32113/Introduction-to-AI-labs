@echo off
echo === Introduction to AI labs - one-time setup ===
python --version || (echo Python not found. Install Python 3.12 and tick "Add to PATH". & pause & exit /b 1)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import textblob" && python -m textblob.download_corpora lite
echo.
echo Setup finished. Now open this folder in VS Code.
pause
