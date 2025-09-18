#!/usr/bin/env python3
"""
Simple HTTP server for serving the EDA project files
"""
import http.server
import socketserver
import os
import sys

def start_server(port=8000):
    """Start a simple HTTP server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"🚀 Server running at http://localhost:{port}")
            print(f"📁 Serving files from: {os.getcwd()}")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use")
            # Try a different port
            start_server(port + 1)
        else:
            print(f"❌ Server error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    start_server(port)
