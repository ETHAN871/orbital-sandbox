"""Dev-only HTTP server: no-cache + state-dump receiver.

Two responsibilities:

1. Subclass SimpleHTTPRequestHandler to prepend Cache-Control: no-store
   to every response, so Chrome doesn't heuristically cache ESM modules
   (which has bitten this codebase repeatedly during development).

2. Accept POST requests to /dump — the page sends a JSON snapshot of
   the physics ring buffer when the user presses D (or clicks the dump
   button). The server writes the body to
   `<server_dir>/.claude/dumps/dump_<ms>.json` so Claude can read it
   via the Read tool and diagnose runtime bugs the user observes.

Used only by the dev preview server (see .claude/launch.json); the
production GitHub Pages deploy is unaffected.

Usage:
  python .claude/nocache_server.py <port> --directory <dir>
"""

import json
import os
import sys
import time
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial


DUMP_SUBDIR = ".claude/dumps"


class ResilientHTTPServer(HTTPServer):
    """HTTPServer hardened for dev-loop survivability.

    - allow_reuse_address bypasses Windows' TIME_WAIT lockout when the
      preview MCP restarts the server quickly (default Python behavior
      can leave the port unbindable for ~30 s after kill, manifesting as
      "Address already in use" on restart).
    - handle_error swallows benign socket-disconnect errors (Chrome
      aggressively closes mid-response during fast reloads). Without
      this, an EPIPE or ECONNRESET during e.g. a hot-reload race could
      propagate up and kill the serve_forever loop, leaving the page
      stranded on chrome-error://.
    """
    allow_reuse_address = True
    # daemon_threads ensures lingering request handlers don't keep the
    # process alive past serve_forever exit. Not strictly required for
    # HTTPServer (single-threaded) but harmless and future-proof if we
    # ever switch to ThreadingHTTPServer.
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc_type, exc_val, _ = sys.exc_info()
        if exc_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return
        print(f"[server-error] {client_address}: {exc_type.__name__}: {exc_val}",
              file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        # Allow the page to POST JSON without a CORS preflight hiccup.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if not self.path.startswith("/dump"):
            self.send_error(404, "Only /dump is a POST endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            # Sanity-check JSON before persisting.
            try:
                json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_error(400, f"Invalid JSON: {e}")
                return

            dump_dir = os.path.join(self.directory, DUMP_SUBDIR)
            os.makedirs(dump_dir, exist_ok=True)
            ts_ms = int(time.time() * 1000)
            filename = f"dump_{ts_ms}.json"
            path = os.path.join(dump_dir, filename)
            with open(path, "wb") as f:
                f.write(body)

            rel = os.path.relpath(path, self.directory).replace("\\", "/")
            print(f"[dump] wrote {rel} ({len(body)} bytes)", flush=True)

            resp = json.dumps({
                "ok": True,
                "path": rel,
                "size": len(body),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except Exception as e:
            self.send_error(500, f"Server error: {e}")


def main():
    port = 8123
    directory = "."
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--directory":
            if i + 1 >= len(args):
                print("Error: --directory requires a path argument", file=sys.stderr)
                sys.exit(1)
            directory = args[i + 1]
            i += 2
        else:
            port = int(args[i])
            i += 1
    handler = partial(NoCacheHandler, directory=directory)
    server = ResilientHTTPServer(("", port), handler)
    print(f"nocache-server listening on http://localhost:{port}/ "
          f"(directory={directory}; POST /dump -> {DUMP_SUBDIR}/) "
          f"[resilient: allow_reuse_address + handle_error guard]",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # Clean exit on Ctrl-C / SIGINT — release the port immediately
        # instead of leaving TIME_WAIT residue (which would have already
        # been mitigated by allow_reuse_address, but explicit shutdown
        # is still cleaner).
        print("\n[shutdown] SIGINT received; closing server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
