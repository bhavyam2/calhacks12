import time, json, threading
import cv2
from ultralytics import YOLO
from flask import Flask, Response
import requests

# Load configuration
with open("config.json") as f:
    cfg = json.load(f)

SERVER = cfg["server_url"].rstrip("/")
CAM_INDEX = cfg.get("camera_index", 1)
ABSENT_THRESHOLD = cfg.get("absent_seconds_threshold", 6)
DEVICE_ID = cfg.get("device_id", "device-1")
LAT = cfg.get("lat")
LON = cfg.get("lon")

# Load YOLO model
model = YOLO("yolov8n.pt")

app = Flask(__name__)
latest_frame = None

def send_theft_alert():
    payload = {
        "device_id": DEVICE_ID,
        "lat": LAT,
        "lon": LON,
        "timestamp": int(time.time())
    }
    try:
        resp = requests.post(SERVER + "/report_theft", json=payload, timeout=5)
        print("Theft reported:", resp.status_code)
    except Exception as e:
        print("Failed to send theft alert:", e)

def camera_loop():
    """Read camera frames, run YOLO detection, and update latest_frame."""
    global latest_frame
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

    absent_since = None
    alerted = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("frame grab failed")
            time.sleep(0.1)
            continue

        # Run YOLO
        results = model(frame, imgsz=640, conf=0.25, verbose=False)
        annotated = results[0].plot()

        # Detect if a person is visible
        labels = [model.names[int(c)] for c in results[0].boxes.cls]
        has_person = "person" in [l.lower() for l in labels]

        if has_person:
            absent_since = None
            alerted = False
        else:
            if absent_since is None:
                absent_since = time.time()
            if (time.time() - absent_since) >= ABSENT_THRESHOLD and not alerted:
                threading.Thread(target=send_theft_alert, daemon=True).start()
                alerted = True

        latest_frame = annotated
        time.sleep(1 / 20.0)  # target 20 FPS

def gen_frames():
    """Yield JPEG frames as MJPEG stream."""
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    print("Camera loop started, visit http://localhost:8009/video_feed")
    app.run(host="0.0.0.0", port=8009, debug=False)
