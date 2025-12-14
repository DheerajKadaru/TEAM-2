#!/bin/bash
echo "Starting SAGAR Backend Server..."
cd "$(dirname "$0")"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

