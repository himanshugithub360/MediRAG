"""Production server for Windows, macOS and Linux (waitress), instead of Flask's dev server.

    python serve.py

Uses HOST and PORT from .env (default 127.0.0.1:8080). Set HOST=0.0.0.0 to allow other
devices on your network. In Docker the app runs under gunicorn instead (see Dockerfile).
"""
import logging

from waitress import serve

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    from app import app, services

    print("Loading models (first start downloads ~180 MB)...")
    try:
        services()
    except RuntimeError as e:
        raise SystemExit(str(e))
    print(f"Dr. Insight is running on http://{'localhost' if config.HOST in ('127.0.0.1', '0.0.0.0') else config.HOST}:{config.PORT}")
    serve(
        app,
        host=config.HOST,
        port=config.PORT,
        threads=8,             # several visitors (and streaming answers) at once
        channel_timeout=300,   # long answers and uploads can take a while
        send_bytes=1,          # flush each streamed word immediately instead of buffering
    )
