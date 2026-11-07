"""WSGI entry point for gunicorn (used in Docker):

    gunicorn wsgi:app

By default the models are loaded here, before the server accepts requests (use with
--preload, as in the main Dockerfile). With BACKGROUND_WARMUP=1 (Dockerfile.render) the port
opens immediately and the models load in a background thread: hosts like Render then see the
app start quickly, /healthz reports "starting" until it's ready, and early requests wait for it.
"""
import logging
import os
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from app import app, services  # noqa: E402


def _warm_up():
    try:
        services()
    except Exception:
        logging.getLogger(__name__).exception("Loading the models failed")


if os.environ.get("BACKGROUND_WARMUP", "").lower() in ("1", "true", "yes"):
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
else:
    services()
