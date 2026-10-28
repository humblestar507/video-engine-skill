"""Serves the business folder and accepts frames: POST /upload/<name>/<index>  (raw PNG bytes or a PNG data URL)."""
import base64
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "frames"


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "upload":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            if body.startswith(b"data:image/png;base64,"):
                body = base64.b64decode(body.split(b",", 1)[1])
            d = OUT / parts[1]
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{int(parts[2]):05d}.png").write_bytes(body)
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8770), H).serve_forever()
