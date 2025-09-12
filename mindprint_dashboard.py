import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import altair as alt
from collections import deque

# WebSocket client to fetch live data
async def fetch_data():
    data_queue = deque(maxlen=60)  # Store last 60 seconds for charting
    async with websockets.connect("ws://localhost:8765") as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            data_queue.append(data)
            yield data_queue  # Yield the queue for live updates
            await asyncio.sleep(1)

# Streamlit app
st.title("Mindprint Brain-Wave Dashboard")
st.write("Live simulated EEG bands. Ensure the server is running!")

# Placeholder for chart
chart_placeholder = st.empty()

# Run the async fetch in Streamlit (using explicit loop)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
data_gen = loop.run_until_complete(fetch_data().__anext__())  # Start generator

while True:
    try:
        data_queue = loop.run_until_complete(data_gen.__anext__())
        if data_queue:
            df = pd.DataFrame(data_queue)
            chart = alt.Chart(df).mark_line().encode(
                x='timestamp:T',
                y=alt.Y('value:Q', scale=alt.Scale(domain=(0, 1))),
                color='band:N'
            ).transform_fold(['delta', 'theta', 'alpha', 'beta', 'gamma'], as_=['band', 'value'])
            chart_placeholder.altair_chart(chart, use_container_width=True)
    except StopIteration:
        break
