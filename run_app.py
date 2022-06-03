import socket
import subprocess
import sys
from pathlib import Path
import webview


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    _, port = sock.getsockname()
    sock.close()
    return port


if __name__ == "__main__":
    file = Path(sys.argv[0]).resolve().parent / "app.py"
    port = get_free_port()
    app = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            file,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    window = webview.create_window(
        "Misaka Writer",
        f"http://127.0.0.1:{port}",
        width=1280,
        height=720,
    )

    def listen():
        while app.poll() is None:
            if app.stdout:
                print(app.stdout.readline().decode("utf-8"), end="")

    webview.start(listen)
    app.terminate()
