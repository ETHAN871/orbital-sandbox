"""Dev-only HTTP server that disables HTTP caching.

Python's stdlib http.server emits Last-Modified but no Cache-Control. Chrome
then heuristically caches JS modules for 10% of (now - Last-Modified), which
means recently-edited files can stay stale in the browser for several minutes
after they're written to disk. For an ESM-no-bundler workflow this corrupts
the dev loop — edits silently no-op until cache lifetime expires.

This wrapper subclasses SimpleHTTPRequestHandler and prepends headers that
tell the browser to revalidate every request. It's used only by the dev
preview server (see .claude/launch.json); the production GitHub Pages deploy
is unaffected.

Usage:
  python .claude/nocache_server.py <port> --directory <dir>
"""

import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


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
    server = HTTPServer(("", port), handler)
    print(f"nocache-server listening on http://localhost:{port}/ "
          f"(directory={directory})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
