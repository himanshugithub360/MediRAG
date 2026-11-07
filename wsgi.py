"""WSGI entry point for gunicorn (used in Docker):

    gunicorn --preload wsgi:app

Models are loaded here, before the server starts accepting requests.
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from app import app, services  # noqa: E402

services()
