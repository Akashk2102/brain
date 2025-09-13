import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import altair as alt
import random
import datetime
from collections import deque
import nest_asyncio
import time
from streamlit_autorefresh import st_autorefresh

# Apply nest_asyncio
nest_asyncio.apply()

# ---------------------------
# Initialize session state
# ---------------------------
if 'data_queue' not in st.session_state:
    st.session_state.data_queue = deque(maxlen=60)
    st.session_state.detected_state = "Initializing..."
    st.session_state.selected_state = 'relaxed'
    st.session_state.previous_state = 'relaxed'
    st.session_state.loop = asyncio.new_event_loop()
    st.session_state.websocket = None
    st.session_state.last_send = 0
    st.session_state.last_recv = 0

if 'dataset' not in st.session_state:
    st.session_state.dataset = None
    st.session_state.generated = False

# ---------------------------
# Fake EEG generator with thought labels
# ---------------------------
def generate_fake_eeg(n=50):
    rows = []
    thoughts = ['hungry', 'thirsty', 'emergency', 'depressed', 'bored', 'relaxed']
    
    for _ in range(n):
        thought = random.choice(thoughts)
        rows.append({
            "thought": thought,
            "ch1": round(random.uniform(0, 1), 3),
            "ch2": round(random.uniform(0, 1), 3),
            "ch3": round(random.uniform(0, 1), 3),
            "ch4": round(random.uniform(0, 1), 3),
            "ch5": round(random.uniform(0, 1), 3),
            "ch6": round(random.uniform(0, 1), 3),
            "ch7": round(random.uniform(0, 1), 3),
            "ch8": round(random.uniform(0, 1), 3),
        })
    return pd.DataFrame(rows)

# ---------------------------
# Function to fetch and update from server
# ---------------------------
def fetch_and_update(force_send=False):
    loop = st.session_state.loop

    if st.session_state.websocket is None:
        try:
            st.session_state.websocket = loop.run_until_complete(
                websockets.connect("ws://localhost:8765", ping_interval=20, ping_timeout=60)
            )
            st.write("Debug: Connected to server.")
        except Exception as e:
            st.error(f"Debug: Connection error - {e}")
            return st.session_state.data_queue

    # SEND
    if force_send or time.time() - st.session_state.last_send >= 1:
        try:
            loop.run_until_complete(
                st.session_state.websocket.send(json.dumps({'state': st.session_state.selected_state}))
            )
            st.write("Debug: State sent to server.")
            st.session_state.last_send = time.time()
        except Exception as e:
            st.error(f"Debug: Send error - {e}")

    # RECEIVE
    if time.time() - st.session_state.last_recv >= 1:
        try:
            message = loop.run_until_complete(
                asyncio.wait_for(st.session_state.websocket.recv(), timeout=2)
            )
            data = json.loads(message)
            st.session_state.data_queue.append(data)
            st.write(f"Debug: Data received and appended - {data}")
            st.session_state.last_recv = time.time()
        except Exception as e:
            st.error(f"Debug: Receive error - {e}")

    return st.session_state.data_queue

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🧠 Mindprint Brain-Wave Dashboard")
st.write("Live simulated EEG bands. Ensure the WebSocket server is running!")

# State selector
states = ['relaxed', 'bored', 'depressed', 'thirsty', 'hungry', 'emergency']
selected_state = st.selectbox("Select Simulated State", states, index=states.index(st.session_state.selected_state))
force_send = (selected_state != st.session_state.previous_state)
st.session_state.selected_state = selected_state
st.session_state.previous_state = selected_state
if force_send:
    st.write(f"Debug: State changed to {selected_state}, forcing send.")

# ---------------------------
# Data Generation Buttons
# ---------------------------
st.subheader("📊 Data Collection & Prediction")

if st.button("💾 Save Data (Training)"):
    df = generate_fake_eeg(100)
    df.to_csv("training_data.csv", index=False)
    st.session_state.dataset = df
    st.session_state.generated = True
    st.success("✅ Training data generated & saved to training_data.csv")
    st.write("📊 Generated data preview:")
    st.dataframe(df.head(10))

if st.button("🔄 Reload ML Model"):
    import requests
    try:
        response = requests.post("http://localhost:5001/api/reload-model")
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Model reloaded successfully! Data shape: {result['data_shape']}")
        else:
            st.error(f"❌ Failed to reload model: {response.text}")
    except Exception as e:
        st.error(f"❌ Error connecting to brainwave API: {e}")

if st.button("▶️ Start Prediction Stream"):
    if st.session_state.generated and st.session_state.dataset is not None:
        df = st.session_state.dataset.sample(frac=1).reset_index(drop=True)
        st.success("Streaming data to brainwave prediction API...")

        import requests
        predictions = []
        
        for _, row in df.iterrows():
            try:
                # Send brainwave data to prediction API
                brainwave_data = {
                    "ch1": row["ch1"],
                    "ch2": row["ch2"], 
                    "ch3": row["ch3"],
                    "ch4": row["ch4"],
                    "ch5": row["ch5"],
                    "ch6": row["ch6"],
                    "ch7": row["ch7"],
                    "ch8": row["ch8"]
                }
                
                response = requests.post("http://localhost:5001/api/predict-thought", 
                                       json=brainwave_data, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_thought = result.get("thought", "unknown")
                    actual_thought = row["thought"]
                    
                    predictions.append({
                        "actual": actual_thought,
                        "predicted": predicted_thought,
                        "correct": actual_thought == predicted_thought
                    })
                    
                    st.write(f"🧠 Actual: {actual_thought} | Predicted: {predicted_thought} | ✅" if actual_thought == predicted_thought else f"🧠 Actual: {actual_thought} | Predicted: {predicted_thought} | ❌")
                else:
                    st.error(f"❌ Prediction failed: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error sending data: {e}")
            
            time.sleep(0.5)
        
        # Show prediction results
        if predictions:
            correct_predictions = sum(1 for p in predictions if p["correct"])
            accuracy = correct_predictions / len(predictions) * 100
            st.success(f"📊 Prediction Complete! Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(predictions)})")
            
            # Show detailed results
            results_df = pd.DataFrame(predictions)
            st.write("📈 Detailed Results:")
            st.dataframe(results_df)
    else:
        st.warning("⚠️ Please generate and save data first before streaming.")

# Real-time prediction mode
st.subheader("🔄 Real-time Prediction Mode")
if st.button("🎯 Start Real-time Prediction"):
    if st.session_state.generated and st.session_state.dataset is not None:
        st.session_state.real_time_prediction = True
        st.success("🎯 Real-time prediction started! Generating and predicting data...")
    else:
        st.warning("⚠️ Please generate training data first.")

if 'real_time_prediction' in st.session_state and st.session_state.real_time_prediction:
    import requests
    
    # Generate new data point
    new_data = generate_fake_eeg(1).iloc[0]
    
    try:
        # Send to prediction API
        brainwave_data = {
            "ch1": new_data["ch1"],
            "ch2": new_data["ch2"], 
            "ch3": new_data["ch3"],
            "ch4": new_data["ch4"],
            "ch5": new_data["ch5"],
            "ch6": new_data["ch6"],
            "ch7": new_data["ch7"],
            "ch8": new_data["ch8"]
        }
        
        response = requests.post("http://localhost:5001/api/predict-thought", 
                               json=brainwave_data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            predicted_thought = result.get("thought", "unknown")
            confidence = result.get("confidence", 0)
            actual_thought = new_data["thought"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actual Thought", actual_thought)
            with col2:
                st.metric("Predicted Thought", predicted_thought)
            with col3:
                st.metric("Confidence", f"{confidence:.2%}")
            
            if actual_thought == predicted_thought:
                st.success("✅ Prediction Correct!")
            else:
                st.error("❌ Prediction Incorrect")
        else:
            st.error(f"❌ Prediction failed: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")

if st.button("⏹️ Stop Real-time Prediction"):
    if 'real_time_prediction' in st.session_state:
        st.session_state.real_time_prediction = False
    st.info("⏹️ Real-time prediction stopped.")

# ---------------------------
# Fetch live data (normal dashboard mode)
# ---------------------------
data_queue = fetch_and_update(force_send=force_send)

if data_queue:
    st.write(f"Debug: Data queue length - {len(st.session_state.data_queue)}")
    df = pd.DataFrame(data_queue)
    chart = alt.Chart(df).mark_line().encode(
        x='timestamp:T',
        y=alt.Y('value:Q', scale=alt.Scale(domain=(0, 1))),
        color='band:N'
    ).transform_fold(['delta', 'theta', 'alpha', 'beta', 'gamma'], as_=['band', 'value'])
    st.altair_chart(chart, use_container_width=True)

    latest = df.iloc[-1]
    if latest['beta'] > 0.5:
        st.session_state.detected_state = "Hungry (High Beta)"
    elif latest['alpha'] > 0.6:
        st.session_state.detected_state = "Thirsty (High Alpha)"
    else:
        st.session_state.detected_state = "Relaxed"
else:
    st.warning("Debug: No data in queue yet. Waiting for server...")

st.write(f"Detected Mindprint State: **{st.session_state.detected_state}**")

# Auto-refresh
st_autorefresh(interval=1000, limit=None, key="data_refresh")