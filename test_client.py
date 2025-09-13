import asyncio
import websockets
import json

async def receive():
    async with websockets.connect("ws://localhost:8765") as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(data)  # Should print data every second

asyncio.run(receive())