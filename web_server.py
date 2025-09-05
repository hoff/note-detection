#!/usr/bin/env python3
"""WebSocket server for streaming real-time note detection to browser."""

import asyncio
import websockets
import json
import http.server
import socketserver
import threading
import os
from pathlib import Path

class NoteStreamServer:
    def __init__(self, websocket_port=8765, http_port=8000):
        self.websocket_port = websocket_port
        self.http_port = http_port
        self.clients = set()
        self.note_buffer = []
        
    async def register_client(self, websocket, path):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast_notes(self, note_data):
        """Broadcast note data to all connected clients."""
        if self.clients:
            message = json.dumps(note_data)
            # Send to all clients
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected

    def add_note_data(self, notes):
        """Add note data to be broadcast (called from main thread)."""
        # This will be called from the note detection script
        asyncio.create_task(self.broadcast_notes(notes))

    def start_http_server(self):
        """Start HTTP server for serving the web interface."""
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)
        
        with socketserver.TCPServer(("", self.http_port), Handler) as httpd:
            print(f"HTTP server serving at http://localhost:{self.http_port}")
            httpd.serve_forever()

    async def start_websocket_server(self):
        """Start the WebSocket server."""
        print(f"WebSocket server starting on ws://localhost:{self.websocket_port}")
        await websockets.serve(self.register_client, "localhost", self.websocket_port)

    def run(self):
        """Run both HTTP and WebSocket servers."""
        # Start HTTP server in a separate thread
        http_thread = threading.Thread(target=self.start_http_server, daemon=True)
        http_thread.start()
        
        # Start WebSocket server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_websocket_server())
        loop.run_forever()

# Global server instance for note detection script to use
note_server = NoteStreamServer()

def send_notes_to_browser(notes):
    """Function to be called from note detection script."""
    if note_server.clients:
        asyncio.create_task(note_server.broadcast_notes(notes))

if __name__ == "__main__":
    print("Starting Note Detection Web Server...")
    print("Open http://localhost:8000 in your browser to see the note visualization")
    note_server.run()