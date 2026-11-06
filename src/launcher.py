"""Open the chat in a browser (default, Chrome or Edge) once the server is ready."""
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

BROWSER_CHOICES = ("default", "chrome", "edge", "none")


def _find_browser(name):
    """Return a command (list) that launches Chrome or Edge, or None if not installed."""
    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        rel = {"chrome": r"Google\Chrome\Application\chrome.exe",
               "edge": r"Microsoft\Edge\Application\msedge.exe"}[name]
        for base in (pf, pf86, local):
            path = os.path.join(base, rel)
            if base and os.path.isfile(path):
                return [path]
    elif sys.platform == "darwin":
        app_name = {"chrome": "Google Chrome", "edge": "Microsoft Edge"}[name]
        if os.path.isdir(f"/Applications/{app_name}.app"):
            return ["open", "-a", app_name]
    else:
        candidates = {"chrome": ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"],
                      "edge": ["microsoft-edge", "microsoft-edge-stable"]}[name]
        for exe in candidates:
            if shutil.which(exe):
                return [exe]
    return None


def open_browser(url, choice):
    if choice in ("chrome", "edge"):
        cmd = _find_browser(choice)
        if cmd:
            subprocess.Popen(cmd + [url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        print(f"{choice.title()} was not found; opening your default browser instead.")
    webbrowser.open(url)


def _open_when_ready(url, choice, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    open_browser(url, choice)


def open_browser_when_ready(url, choice):
    """Wait (in a background thread) until the server answers, then open the browser."""
    threading.Thread(target=_open_when_ready, args=(url, choice), daemon=True).start()
