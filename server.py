#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='.', **kwargs)

def main():
    port = int(os.environ.get("PORT", 10000))
    
    print(f"🚀 Starting server on 0.0.0.0:{port}")
    print(f"📁 Current directory: {os.getcwd()}")
    
    # List available files
    try:
        files = [f for f in os.listdir('.') if f.endswith(('.html', '.txt', '.png', '.py'))]
        print(f"📋 Available files: {files}")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    # Create server
    try:
        with socketserver.TCPServer(("0.0.0.0", port), MyHandler) as httpd:
            print(f"✅ Server successfully bound to port {port}")
            print(f"🌐 Server is ready to receive requests")
            httpd.serve_forever()
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
