import sys
import os

# Make sure the project root is in path so `app` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mangum import Mangum
from app.main import app

# Mangum wraps the FastAPI ASGI app for AWS Lambda / Netlify Functions
handler = Mangum(app, lifespan="off")
