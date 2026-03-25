@echo off
cd /d "%~dp0\backend"
call ..\.venv\Scripts\activate.bat
python -m streamlit run UI.py --server.port 8502
