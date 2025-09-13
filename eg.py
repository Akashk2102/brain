from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
import time

app = Flask(__name__)
CORS(app)

# Video streaming route and generator
def gen_frames():
    while True:
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw quadrants and options
        h, w, _ = frame.shape
        option_names = ["Food", "Water", "Emergency", "Songs"]
        colors = [(255, 255, 255)] * 4
        # Get focus and selected from latest_tracking_data
        focus_idx = latest_tracking_data.get('focus_index')
        selected_option = latest_tracking_data.get('selected_option')
        for i, name in enumerate(option_names):
            if focus_idx == i:
                colors[i] = (0,255,255)  # Focused: yellow
            if selected_option == name:
                colors[i] = (0, 0, 255)    # Selected: green

        # Draw rectangles for quadrants
        cv2.rectangle(frame, (0, 0), (w//2, h//2), colors[0], 4)
        cv2.rectangle(frame, (w//2, 0), (w, h//2), colors[1], 4)
        cv2.rectangle(frame, (0, h//2), (w//2, h), colors[2], 4)
        cv2.rectangle(frame, (w//2, h//2), (w, h), colors[3], 4)

        # Put option names in each quadrant
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        cv2.putText(frame, option_names[0], (30, 50), font, font_scale, colors[0], thickness, cv2.LINE_AA)
        cv2.putText(frame, option_names[1], (w//2 + 30, 50), font, font_scale, colors[1], thickness, cv2.LINE_AA)
        cv2.putText(frame, option_names[2], (30, h//2 + 50), font, font_scale, colors[2], thickness, cv2.LINE_AA)
        cv2.putText(frame, option_names[3], (w//2 + 30, h//2 + 50), font, font_scale, colors[3], thickness, cv2.LINE_AA)

        # Optionally, show a message for selected option
        if selected_option:
            cv2.putText(frame, f"Selected: {selected_option}", (w//2 - 180, h - 30), font, 1.2, (0, 200, 0), 4, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Mediapipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
cap = cv2.VideoCapture(0)

# Constants
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.2
SELECTION_DELAY = 3  # seconds
options = ["Food", "Water", "Emergency", "Songs"]

# State variables
ear_queue = deque(maxlen=5)
gaze_queue = deque(maxlen=15)
eye_closed_start_time = None

calib_left_x = []
calib_left_y = []
calib_right_x = []
calib_right_y = []
calibrated = False
calib_frames = 150
frame_count = 0
left_min_x = left_max_x = left_min_y = left_max_y = None
right_min_x = right_max_x = right_min_y = right_max_y = None

eye_tracking_enabled = False

latest_tracking_data = {
    'gx': 0.5,
    'gy': 0.5,
    'focus_index': None,
    'eyes_closed': None,
    'selected_option': None
}


def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    h, w, _ = frame_shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)


def eye_center(landmarks, eye_indices, frame_shape):
    h, w, _ = frame_shape
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    cx = np.mean([p[0] for p in points])
    cy = np.mean([p[1] for p in points])
    return cx, cy


def get_face_bbox(landmarks, frame_shape):
    h, w, _ = frame_shape
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def eye_tracking_loop():
    global frame_count, calibrated
    global left_min_x, left_max_x, left_min_y, left_max_y
    global right_min_x, right_max_x, right_min_y, right_max_y
    global eye_closed_start_time, latest_tracking_data

    while True:
        if not eye_tracking_enabled:
            latest_tracking_data = {
                'gx': 0.5,
                'gy': 0.5,
                'focus_index': None,
                'eyes_closed': None,
                'selected_option': None
            }
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(bgr)

        if not results.multi_face_landmarks:
            time.sleep(0.1)
            continue

        landmarks = results.multi_face_landmarks[0]

        right_ear = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE, frame.shape)
        left_ear = eye_aspect_ratio(landmarks.landmark, LEFT_EYE, frame.shape)
        avg_ear = (right_ear + left_ear) / 2
        ear_queue.append(avg_ear)
        smooth_ear = np.mean(ear_queue)
        eyes_closed = smooth_ear < BLINK_THRESHOLD

        # Calibration phase
        if not calibrated:
            rx, ry = eye_center(landmarks.landmark, RIGHT_EYE, frame.shape)
            lx, ly = eye_center(landmarks.landmark, LEFT_EYE, frame.shape)
            calib_left_x.append(lx)
            calib_left_y.append(ly)
            calib_right_x.append(rx)
            calib_right_y.append(ry)
            frame_count += 1

            if frame_count >= calib_frames:
                left_min_x, left_max_x = min(calib_left_x), max(calib_left_x)
                left_min_y, left_max_y = min(calib_left_y), max(calib_left_y)
                right_min_x, right_max_x = min(calib_right_x), max(calib_right_x)
                right_min_y, right_max_y = min(calib_right_y), max(calib_right_y)
                calibrated = True

            time.sleep(0.1)
            continue

        rx, ry = eye_center(landmarks.landmark, RIGHT_EYE, frame.shape)
        lx, ly = eye_center(landmarks.landmark, LEFT_EYE, frame.shape)

        rx_norm = (rx - right_min_x) / (right_max_x - right_min_x + 1e-6)
        ry_norm = (ry - right_min_y) / (right_max_y - right_min_y + 1e-6)
        lx_norm = (lx - left_min_x) / (left_max_x - left_min_x + 1e-6)
        ly_norm = (ly - left_min_y) / (left_max_y - left_min_y + 1e-6)

        rx_norm = np.clip(rx_norm, 0, 1)
        ry_norm = np.clip(ry_norm, 0, 1)
        lx_norm = np.clip(lx_norm, 0, 1)
        ly_norm = np.clip(ly_norm, 0, 1)

        gx_raw = (rx_norm + lx_norm) / 2.0
        gy_raw = (ry_norm + ly_norm) / 2.0
        gx = 1.0 - gx_raw
        gy = gy_raw

        gaze_queue.append((gx, gy))
        smooth_gx = np.mean([g[0] for g in gaze_queue])
        smooth_gy = np.mean([g[1] for g in gaze_queue])

        face_x1, face_y1, face_x2, face_y2 = get_face_bbox(landmarks.landmark, frame.shape)
        face_w = face_x2 - face_x1
        face_h = face_y2 - face_y1
        gaze_x_rel = (gx * w - face_x1) / (face_w + 1e-6)
        gaze_y_rel = (gy * h - face_y1) / (face_h + 1e-6)
        gaze_x_rel = np.clip(gaze_x_rel, 0, 1)
        gaze_y_rel = np.clip(gaze_y_rel, 0, 1)

        focus_index = None
        if smooth_gx > 0.5 and smooth_gy < 0.5:
            focus_index = 0  # Food
        elif smooth_gx <= 0.5 and smooth_gy < 0.5:
            focus_index = 1  # Water
        elif smooth_gx > 0.5 and smooth_gy >= 0.5:
            focus_index = 2  # Emergency
        else:
            focus_index = 3  # Songs

        selected_option = None
        if eyes_closed:
            global eye_closed_start_time
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            elapsed = time.time() - eye_closed_start_time
            if elapsed >= SELECTION_DELAY:
                selected_option = options[focus_index]
        else:
            eye_closed_start_time = None

        latest_tracking_data = {
            'gx': float(smooth_gx),
            'gy': float(smooth_gy),
            'focus_index': int(focus_index) if focus_index is not None else None,
            'eyes_closed': bool(eyes_closed),
            'selected_option': selected_option
        }

        time.sleep(0.1)


@app.route('/api/eye-tracking')
def get_eye_tracking_data():
    return jsonify(latest_tracking_data)


@app.route('/api/eye-tracking/start', methods=['POST'])
def start_eye_tracking():
    global eye_tracking_enabled
    eye_tracking_enabled = True
    return jsonify({"status": "Eye tracking started"})


@app.route('/api/eye-tracking/stop', methods=['POST'])
def stop_eye_tracking():
    global eye_tracking_enabled
    eye_tracking_enabled = False
    return jsonify({"status": "Eye tracking stopped"})


if __name__ == '__main__':
    threading.Thread(target=eye_tracking_loop, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)