@echo off
echo Starting Crew: Autonomous AI Classroom Builder (Omi x Qdrant x Lyzr)...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
