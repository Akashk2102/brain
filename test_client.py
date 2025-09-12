import asyncio
import websockets
import json

async def receive():
    async with websockets.connect("ws://localhost:8765") as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(data)

# Run with explicit loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(receive())
loop.close()
