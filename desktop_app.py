from __future__ import annotations

import threading
import time

from accreditx_app import app


def main():
    try:
        import webview
    except ImportError as exc:
        raise SystemExit("Install pywebview first: pip install pywebview") from exc
    thread = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False), daemon=True)
    thread.start(); time.sleep(1)
    webview.create_window("Accredix", "http://127.0.0.1:5000", width=1440, height=940, min_size=(1100, 720))
    webview.start()


if __name__ == "__main__": main()
