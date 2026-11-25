"""Try tiny-ja-wakeword live with your microphone in a browser.

    python -m tiny_ja_wakeword.demo

Then open http://127.0.0.1:8770. Audio is processed locally on this machine and never stored.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .listener import Listener

PAGE = Path(__file__).with_name("page.html")
LIMIT = 4 << 20  # 4 MiB max per request


class Handler(BaseHTTPRequestHandler):
    listener: Listener | None = None

    def log_message(self, *args):
        pass

    def reply(self, body, status: int = 200, kind: str = "application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def body(self, kind: str) -> bytes:
        if self.headers.get_content_type() != kind:
            raise ValueError(f"expected {kind}")
        size = int(self.headers.get("Content-Length", 0))
        if not 0 <= size <= LIMIT:
            raise ValueError("request too large")
        return self.rfile.read(size)

    def do_GET(self):
        if self.path == "/":
            self.reply(PAGE.read_bytes(), kind="text/html")
        elif self.path == "/state":
            assert self.listener is not None
            with self.listener.lock:
                state = self.listener.state()
            self.reply(state)
        else:
            self.reply(dict(error="not found"), 404)

    def do_POST(self):
        assert self.listener is not None
        try:
            path, _, _ = self.path.partition("?")
            if path == "/audio":
                pcm = np.frombuffer(self.body("application/octet-stream"), dtype="<i2")
                self.reply(self.listener.feed(pcm.astype(np.float32) / 32768.0))
            elif path == "/start":
                rate = int(json.loads(self.body("application/json"))["rate"])
                if not 8000 <= rate <= 192000:
                    raise ValueError("マイクの標本化周波数が対応範囲の外です。")
                self.listener.restart(rate)
                self.reply(dict(ok=True))
            elif path == "/configure":
                wanted = json.loads(self.body("application/json"))
                state = self.listener.configure(
                    **{k: wanted[k] for k in ("phrase", "model", "sure", "maybe") if k in wanted}
                )
                self.reply(state)
            else:
                self.reply(dict(error="not found"), 404)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.reply(dict(error=str(error)), 400)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--port", type=int, default=8770)
    cli.add_argument("--keyword", default="オーケー、グーグル", help="初期ウェイクワード（ひらがな・カタカナ）")
    cli.add_argument("--model", default="crnn", help="モデル名（crnn / conv）または ONNX ファイルパス")
    cli.add_argument("--threshold", type=float, default=0.50, help="確定の確率しきい値（0.05〜0.99、標準 0.50）")
    cli.add_argument("--maybe", type=float, default=0.30, help="「かも」の確率しきい値（0.05〜threshold、既定 0.30）")
    cli.add_argument("--open", action="store_true", help="起動時に自動でブラウザを開く")
    args = cli.parse_args()

    Handler.listener = Listener(args.model, args.keyword, args.threshold, args.maybe)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"tiny-ja-wakeword デモ起動中: {url}  （止めるには Ctrl+C）", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
