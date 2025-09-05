#!/usr/bin/env python3
"""Simple script to start the web-based note detection demo."""

import subprocess
import threading
import time
import http.server
import socketserver
import os
import webbrowser

def start_http_server():
    """Start HTTP server for the web interface."""
    PORT = 8000
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"HTTP server serving at http://localhost:{PORT}")
        print(f"Open http://localhost:{PORT} in your browser to see the visualization")
        httpd.serve_forever()

def main():
    print("🎵 Starting Web-based Note Detection Demo 🎵")
    print("=" * 50)
    
    # Start HTTP server in background thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Open browser automatically
    try:
        webbrowser.open('http://localhost:8000')
        print("✅ Browser opened automatically")
    except:
        print("❗ Could not open browser automatically")
        print("Please open http://localhost:8000 manually")
    
    print("\n📋 Instructions:")
    print("1. The web interface should open in your browser")
    print("2. Starting note detection in 3 seconds...")
    print("3. Play music or sing to see real-time note detection!")
    print("4. Press Ctrl+C to stop\n")
    
    time.sleep(3)
    
    # Start the realtime note detection
    try:
        cmd = [
            'python', 'realtime_web.py', 
            '--model_path', 'onsets_frames_wavinput_no_offset_uni.tflite',
            '--web_output'
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTrying fallback command...")
        print("Run this manually if needed:")
        print("python realtime_web.py --model_path onsets_frames_wavinput_no_offset_uni.tflite")

if __name__ == "__main__":
    main()