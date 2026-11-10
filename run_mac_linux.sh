#!/usr/bin/env bash
# Usage: ./run_mac_linux.sh          -> http://localhost:8501
#        ./run_mac_linux.sh phone    -> also reachable from your phone on the same Wi-Fi
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
if [ "$1" = "phone" ]; then
  echo "Open http://$(hostname -I 2>/dev/null | awk '{print $1}' || ipconfig getifaddr en0):8501 on your phone"
  streamlit run app.py --server.address 0.0.0.0
else
  streamlit run app.py
fi
