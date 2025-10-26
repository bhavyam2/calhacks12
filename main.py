import os
import time
import threading
import uuid
import atexit
import ssl
import smtplib
import requests
from datetime import datetime
from email.message import EmailMessage
from flask import Flask, render_template, Response, jsonify, send_file
import cv2
from ultralytics import YOLO

# New dependency for SSH:
import paramiko

# ---------------- CONFIG (env vars) ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11m.pt")
CAPTURE_DIR = os.getenv("CAPTURE_DIR", "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Camera & perf
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "320"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "240"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))

# Detection tuning
TARGET_CLASS = os.getenv("TARGET_CLASS", "suitcase").lower()
PERSON_CLASS = os.getenv("PERSON_CLASS", "person").lower()
MOVE_THRESH = float(os.getenv("MOVE_THRESH", "50"))
MISSING_FRAMES_LIMIT = int(os.getenv("MISSING_FRAMES_LIMIT", "15"))
PERSON_CAPTURE_COOLDOWN = float(os.getenv("PERSON_CAPTURE_COOLDOWN", "5.0"))
IMG_QUALITY = int(os.getenv("IMG_QUALITY", "70"))

# Email config (required to actually email)
EMAIL_ALERTS_ENABLED = os.getenv("EMAIL_ALERTS_ENABLED", "true").lower() in ("1", "true", "yes")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "usb163016@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "xwta yprh zluo gfmm")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "bhadauriya@ucdavis.edu")

# Claude (Anthropic) optional config
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")

# LiveKit optional config
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_ENABLED = bool(LIVEKIT_URL)

# Misc
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://localhost:8010")

# ---------------- SSH CONFIG ----------------
# Remote host to call when suitcase is stolen. Either password auth or key auth supported.
REMOTE_HOST = os.getenv("REMOTE_HOST", "10.42.0.1")            # e.g. "192.168.1.100" or "remote.example.com"
REMOTE_PORT = int(os.getenv("REMOTE_PORT", "22"))
REMOTE_USER = os.getenv("REMOTE_USER", "watchdog")            # e.g. "pi"
REMOTE_PASSWORD = os.getenv("REMOTE_PASSWORD", "123456")    # optional, if using key auth leave blank
REMOTE_KEY_PATH = os.getenv("REMOTE_KEY_PATH", "")    # optional path to private key, e.g. "/home/user/.ssh/id_rsa"
REMOTE_COMMAND = os.getenv("REMOTE_COMMAND", "python3 /home/watchdog/buzzer_test.py")      # command to run remotely, e.g. "bash /home/pi/handle_theft.sh"
SSH_CONNECT_TIMEOUT = float(os.getenv("SSH_CONNECT_TIMEOUT", "10.0"))

# ---------------- Initialize app, camera, model ----------------
app = Flask(__name__)

# single camera capture (AVFoundation for macOS); adjust if on other platform
camera = cv2.VideoCapture("http://10.42.0.1:8080/?action=stream")
#camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not camera.isOpened():
    raise RuntimeError(f"Could not open webcam index {CAMERA_INDEX}; check permissions/index")

# load YOLO model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Failed to load model:", e)
    raise

# ---------------- Shared state ----------------
system_armed = False
status = "System Disarmed"
status_lock = threading.Lock()

latest_frame = None
latest_frame_lock = threading.Lock()

last_suitcase_center = None
missing_frames = 0

last_person_capture_time = 0.0
person_photos = {}  # filename -> metadata
person_photos_lock = threading.Lock()

emailed_for_current_stolen = False
emailed_lock = threading.Lock()

# ---------------- Helpers ----------------
def update_status(new_status, frame_for_snapshot=None):
    """
    Update status; on transition to '🔴 Stolen' save snapshot, send email, and optionally run remote SSH command.
    """
    global status, emailed_for_current_stolen
    with status_lock:
        old = status
        if new_status != old:
            status = new_status
            print(f"[STATUS] {old} -> {new_status}")
            # entering stolen
            if new_status == "🔴 Stolen":
                with emailed_lock:
                    if not emailed_for_current_stolen:
                        emailed_for_current_stolen = True
                        # save snapshot and send email in background
                        fpath = None
                        if frame_for_snapshot is not None:
                            fpath = save_stolen_snapshot(frame_for_snapshot)
                        else:
                            with latest_frame_lock:
                                lf = None if latest_frame is None else latest_frame.copy()
                            if lf is not None:
                                fpath = save_stolen_snapshot(lf)
                        if EMAIL_ALERTS_ENABLED and fpath:
                            threading.Thread(target=send_stolen_email, args=(fpath,), daemon=True).start()

                        # Run SSH command (non-blocking)
                        if REMOTE_HOST and REMOTE_USER and REMOTE_COMMAND:
                            threading.Thread(target=run_ssh_command, args=(REMOTE_HOST, REMOTE_PORT, REMOTE_USER, REMOTE_PASSWORD, REMOTE_KEY_PATH, REMOTE_COMMAND), daemon=True).start()
                        else:
                            print("[SSH] Remote SSH not executed: REMOTE_HOST/REMOTE_USER/REMOTE_COMMAND not configured.")
            # leaving stolen
            if old == "🔴 Stolen" and new_status in ("🟢 Moving", "🟡 Standing still"):
                with emailed_lock:
                    emailed_for_current_stolen = False
                delete_photos_kept_due_to_stolen()

def center_of_box(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def save_person_photo(frame_bgr):
    """Save grayscale person photo (throttled) and schedule evaluation after 10s."""
    global last_person_capture_time
    now = time.time()
    if now - last_person_capture_time < PERSON_CAPTURE_COOLDOWN:
        return None
    last_person_capture_time = now

    filename = f"person_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(CAPTURE_DIR, filename)
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, gray, [cv2.IMWRITE_JPEG_QUALITY, IMG_QUALITY])
    except Exception as e:
        print("[SAVE] Failed to save person photo:", e)
        return None

    meta = {"path": path, "timestamp": now, "kept_due_to_stolen": False}
    with person_photos_lock:
        person_photos[filename] = meta

    t = threading.Timer(10.0, evaluate_and_maybe_delete_photo, args=(filename,))
    t.daemon = True
    t.start()
    print(f"[CAPTURE] Saved grayscale person photo {filename}")
    return path

def evaluate_and_maybe_delete_photo(filename):
    with person_photos_lock:
        meta = person_photos.get(filename)
        if not meta:
            return
    with status_lock:
        cur = status
    if cur in ("🟢 Moving", "🟡 Standing still"):
        try:
            os.remove(meta["path"])
            print(f"[DELETE] Auto-deleted person photo {filename} (status {cur})")
        except FileNotFoundError:
            pass
        with person_photos_lock:
            person_photos.pop(filename, None)
    elif cur == "🔴 Stolen":
        with person_photos_lock:
            person_photos[filename]["kept_due_to_stolen"] = True
        print(f"[KEEP] Photo {filename} kept because status=stolen")
    else:
        try:
            os.remove(meta["path"])
        except FileNotFoundError:
            pass
        with person_photos_lock:
            person_photos.pop(filename, None)

def delete_photos_kept_due_to_stolen():
    to_delete = []
    with person_photos_lock:
        for fname, meta in list(person_photos.items()):
            if meta.get("kept_due_to_stolen"):
                to_delete.append((fname, meta["path"]))
    for fname, path in to_delete:
        try:
            os.remove(path)
            print(f"[DELETE-RECOVERY] Deleted previously-kept photo {fname}")
        except FileNotFoundError:
            pass
        with person_photos_lock:
            person_photos.pop(fname, None)

def save_stolen_snapshot(frame_bgr):
    """Save the snapshot to attach to email when stolen is detected."""
    filename = f"stolen_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(CAPTURE_DIR, filename)
    try:
        cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, IMG_QUALITY])
        print(f"[SNAPSHOT] Saved stolen snapshot: {path}")
        return path
    except Exception as e:
        print("[SNAPSHOT] Failed to save stolen snapshot:", e)
        return None

# ---------------- SSH Helper ----------------
def run_ssh_command(host, port, user, password, key_path, command):
    """
    Connects to remote host via SSH and runs the provided command.
    Uses password auth if password provided, otherwise attempts private key if key_path provided.
    This runs in a background thread so it should not block.
    """
    try:
        print(f"[SSH] Connecting to {user}@{host}:{port} (timeout={SSH_CONNECT_TIMEOUT}s)...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {
            "hostname": host,
            "port": port,
            "username": user,
            "timeout": SSH_CONNECT_TIMEOUT,
            "banner_timeout": SSH_CONNECT_TIMEOUT,
            "auth_timeout": SSH_CONNECT_TIMEOUT,
        }
        if password:
            connect_kwargs["password"] = password
        elif key_path:
            connect_kwargs["key_filename"] = key_path
        else:
            # attempt to use default ssh agent / keys if present
            connect_kwargs["allow_agent"] = True
            connect_kwargs["look_for_keys"] = True

        ssh.connect(**connect_kwargs)
        print(f"[SSH] Connected. Executing remote command: {command}")
        stdin, stdout, stderr = ssh.exec_command(command, timeout=SSH_CONNECT_TIMEOUT)

        # Optionally wait for command to finish and capture output
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if out:
            print("[SSH][out]", out.strip())
        if err:
            print("[SSH][err]", err.strip())

        # close connection
        ssh.close()
        print("[SSH] Remote command finished and connection closed.")
    except Exception as e:
        print(f"[SSH] Error while running remote command: {e}")

# ---------------- Claude integration ----------------
def generate_incident_report_with_claude(snapshot_path=None, status_text=None):
    fallback = f"Suitcase theft alert detected. Status: {status_text or '🔴 Stolen'}."
    if snapshot_path:
        fallback += f" Snapshot saved at: {snapshot_path}"

    if not CLAUDE_API_URL or not CLAUDE_API_KEY:
        print("[CLAUDE] Not configured; using fallback report.")
        return fallback

    prompt = (
        "You are an incident report assistant. Generate a short, factual incident report (3-6 sentences) "
        "for a suitcase security system. Include time (UTC), status, and suggested next steps for the owner. "
        f"Status: {status_text or '🔴 Stolen'}. Snapshot path: {snapshot_path or 'N/A'}.\n\nReport:"
    )

    try:
        headers = {"Authorization": f"Bearer {CLAUDE_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "max_tokens_to_sample": 300
        }
        r = requests.post(CLAUDE_API_URL, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        text = None
        for key in ("completion", "text", "message", "response", "output"):
            if isinstance(data.get(key), str):
                text = data.get(key)
                break
        if not text:
            if "completion" in data:
                text = data["completion"]
            elif "choices" in data and isinstance(data["choices"], list) and "text" in data["choices"][0]:
                text = data["choices"][0]["text"]
        if not text:
            text = str(data)
        print("[CLAUDE] Generated incident report via API.")
        return text
    except Exception as e:
        print("[CLAUDE] Error calling Claude API:", e)
        return fallback

# ---------------- Email sending (includes report + LiveKit link if configured) ----------------
def send_stolen_email(image_path):
    if not EMAIL_ALERTS_ENABLED:
        print("[EMAIL] Email alerts disabled; skipping send.")
        return
    if not SMTP_SERVER or not SMTP_USER or not SMTP_PASSWORD or not EMAIL_TO:
        print("[EMAIL] SMTP not configured correctly; skipping send.")
        return

    with status_lock:
        st = status
    report_text = generate_incident_report_with_claude(snapshot_path=image_path, status_text=st)

    msg = EmailMessage()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    msg["Subject"] = f"[ALERT] Suitcase possibly stolen — {ts}"
    msg["From"] = EMAIL_FROM or SMTP_USER
    msg["To"] = EMAIL_TO
    body_lines = [
        f"Suitcase security alert detected at {ts}.",
        "",
        "Incident report:",
        report_text,
        ""
    ]
    if LIVEKIT_ENABLED:
        body_lines.extend(["Live view link:", LIVEKIT_URL, ""])
    body_lines.append(f"View server captures: {SERVER_BASE_URL}/captures")
    msg.set_content("\n".join(body_lines))

    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=os.path.basename(image_path))
    except Exception as e:
        print("[EMAIL] Failed to attach image:", e)

    try:
        if SMTP_PORT == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        print("[EMAIL] Sent stolen alert email with attachment:", image_path)
    except Exception as e:
        print("[EMAIL] Failed to send email:", e)

# ---------------- Capture + Detection threads ----------------
def capture_thread():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.05)
            continue
        with latest_frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.02)

def detection_thread():
    global last_suitcase_center, missing_frames
    last_seen_time = time.time()
    while True:
        if not system_armed:
            time.sleep(0.1)
            continue
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.02)
            continue

        try:
            results = model(frame, imgsz=320, conf=0.45, verbose=False)
        except Exception as e:
            print("[INFER] Model error:", e)
            time.sleep(0.1)
            continue

        det = results[0]
        boxes = []
        classes = []
        confs = []
        if hasattr(det, "boxes"):
            try:
                boxes = det.boxes.xyxy.cpu().numpy()
                classes = det.boxes.cls.cpu().numpy()
                confs = det.boxes.conf.cpu().numpy()
            except Exception:
                pass
        names = det.names if hasattr(det, "names") else {}

        suitcase_box = None
        person_detected = False

        for box, cls_idx, conf in zip(boxes, classes, confs):
            label = names[int(cls_idx)].lower()
            x1, y1, x2, y2 = map(int, box)
            color = (0,255,0) if label == PERSON_CLASS else (255,165,0)
            try:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception:
                pass
            if label == TARGET_CLASS:
                suitcase_box = (x1, y1, x2, y2)
            if label == PERSON_CLASS:
                person_detected = True

        # suitcase state logic
        if suitcase_box is not None:
            cx, cy = center_of_box(suitcase_box)
            if last_suitcase_center is None:
                update_status("🟡 Standing still")
            else:
                dist = ((cx - last_suitcase_center[0])**2 + (cy - last_suitcase_center[1])**2) ** 0.5
                if dist > MOVE_THRESH:
                    update_status("🟢 Moving")
                else:
                    update_status("🟡 Standing still")
            last_suitcase_center = (cx, cy)
            missing_frames = 0
        else:
            missing_frames += 1
            if missing_frames > MISSING_FRAMES_LIMIT:
                update_status("🔴 Stolen", frame_for_snapshot=frame)

        # person capture
        if person_detected:
            try:
                save_person_photo(frame.copy())
            except Exception as e:
                print("[CAPTURE] error saving person photo:", e)

        time.sleep(0.02)

# ---------------- MJPEG stream + endpoints ----------------
def frame_generator():
    while True:
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.03)
            continue
        with status_lock:
            cur = status
        color = (0,255,0)
        if cur == "🔴 Stolen":
            color = (0,0,255)
        elif cur == "🟡 Standing still":
            color = (0,200,200)
        elif cur == "🟢 Moving":
            color = (0,255,0)
        else:
            color = (200,200,200)
        try:
            cv2.putText(frame, f"Status: {cur}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print("[STREAM] encode error:", e)
            time.sleep(0.03)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def get_status():
    with status_lock:
        return jsonify({"status": status, "armed": system_armed})

@app.route("/captures")
def list_captures():
    out = []
    with person_photos_lock:
        for fname, meta in person_photos.items():
            if meta.get("kept_due_to_stolen"):
                out.append({
                    "filename": fname,
                    "path": f"/captures/download/{fname}",
                    "timestamp": meta.get("timestamp")
                })
    stolen_files = [f for f in os.listdir(CAPTURE_DIR) if f.startswith("stolen_")]
    for f in stolen_files:
        out.append({
            "filename": f,
            "path": f"/captures/download/{f}",
            "timestamp": None
        })
    return jsonify(out)

@app.route("/captures/download/<filename>")
def download_capture(filename):
    path = os.path.join(CAPTURE_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, as_attachment=True)

@app.route("/arm", methods=["POST"])
def arm_system():
    global system_armed, last_suitcase_center, missing_frames
    system_armed = True
    last_suitcase_center = None
    missing_frames = 0
    update_status("🟡 Standing still")
    return jsonify({"status": status, "armed": system_armed})

@app.route("/disarm", methods=["POST"])
def disarm_system():
    global system_armed
    system_armed = False
    update_status("System Disarmed")
    return jsonify({"status": status, "armed": system_armed})

# ---------------- Cleanup & startup ----------------
def cleanup():
    try:
        if camera and camera.isOpened():
            camera.release()
            print("[CLEANUP] Released camera")
    except Exception as e:
        print("[CLEANUP] Error releasing camera:", e)
atexit.register(cleanup)

if __name__ == "__main__":
    # start threads
    tcap = threading.Thread(target=capture_thread, daemon=True)
    tcap.start()
    tdet = threading.Thread(target=detection_thread, daemon=True)
    tdet.start()
    print("Server running on http://127.0.0.1:8010")
    app.run(host="0.0.0.0", port=8010, debug=False)
