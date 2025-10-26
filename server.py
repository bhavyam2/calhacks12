import time, json
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import chromadb
from chromadb import PersistentClient

# --- Flask & SocketIO Setup ---
app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'dev-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# --- ChromaDB Setup ---
client = PersistentClient(path=".chromadb_db")
COLLECTION_NAME = "suitcase_thefts"

try:
    collection = client.get_collection(COLLECTION_NAME)
except:
    collection = client.create_collection(name=COLLECTION_NAME)

# --- Routes ---
@app.route("/")
def index():
    return render_template("templates/index.html")

@app.route("/report_theft", methods=["POST"])
def report_theft():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json"}), 400

    device_id = data.get("device_id", "device-1")
    lat = data.get("lat")
    lon = data.get("lon")
    timestamp = data.get("timestamp", int(time.time()))

    if lat is None or lon is None:
        return jsonify({"error": "lat/lon required"}), 400

    # Store in Chroma
    id_str = f"{device_id}-{timestamp}"
    vector = [float(lat), float(lon)]
    collection.add(
        ids=[id_str],
        embeddings=[vector],
        metadatas=[{"device_id": device_id, "timestamp": timestamp}],
    )

    # Emit to all web clients in real-time
    payload = {"type": "status", "message": f"⚠️ Theft detected near {lat}, {lon}"}
    socketio.emit("new_theft", payload, broadcast=True)
    print("Broadcasted theft event:", payload)

    return jsonify({"status": "ok"}), 200

@app.route("/events", methods=["GET"])
def events():
    try:
        res = collection.get(include=["embeddings", "metadatas"])
        items = []
        for id_, emb, meta in zip(res["ids"], res["embeddings"], res["metadatas"]):
            items.append({
                "id": id_,
                "lat": emb[0],
                "lon": emb[1],
                "metadata": meta
            })
        return jsonify(items)
    except Exception as e:
        print("Error fetching events:", e)
        return jsonify([])

# MJPEG proxy stream (from detector)
@app.route("/video_feed")
def video_feed():
    import requests
    from flask import stream_with_context

    def generate():
        with requests.get("http://localhost:8009/video_feed", stream=True) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Socket Handlers ---
@socketio.on("connect")
def on_connect():
    print("Client connected via websocket")
    emit("status", {"type": "status", "message": "Connected to Flask server"})

# --- Run App ---
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8010)
