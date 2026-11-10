@echo off
REM Usage: run_windows.bat          -> open at http://localhost:8501
REM        run_windows.bat phone    -> also reachable from your phone on the same Wi-Fi
cd /d "%~dp0"
if not exist .venv ( py -3.11 -m venv .venv || py -3 -m venv .venv )
call .venv\Scripts\activate.bat
pip install -r requirements-local.txt
if /I "%1"=="phone" (
  echo Open  http://YOUR-PC-IP:8501  on your phone ^(find the IP with: ipconfig^)
  streamlit run app.py --server.address 0.0.0.0
) else (
  streamlit run app.py
)
