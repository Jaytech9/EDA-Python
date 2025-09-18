import http.server
import socketserver
import os

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='.', **kwargs)

port = int(os.environ.get("PORT", 10000))
with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
    print(f"Server running on port {port}")
    httpd.serve_forever()
