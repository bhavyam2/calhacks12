from flask import Flask, jsonify, request
from flask_cors import CORS  # ✅ Add this import
from chromadb import Client
import json
import os
import hashlib

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Path to JSON file
JSON_PATH = "data/points.json"
CHROMA_COLLECTION_NAME = "points"

# Initialize ChromaDB client
client = Client()
collection = client.create_collection(CHROMA_COLLECTION_NAME)

# Load points from JSON file on startup
def load_from_json():
    if not os.path.exists(JSON_PATH):
        print("✅ JSON file not found. Creating empty one.")
        return []

    with open(JSON_PATH, "r") as f:
        try:
            data = json.load(f)
            print(f"✅ Loaded {len(data)} points from {JSON_PATH}")
            return data
        except Exception as e:
            print("❌ Error loading JSON:", e)
            return []

# Save points to JSON file
def save_to_json(points):
    with open(JSON_PATH, "w") as f:
        json.dump(points, f, indent=2)
    print(f"✅ Saved {len(points)} points to {JSON_PATH}")

# Add a point to ChromaDB and JSON
def add_point(lat, lng, intensity):
    vector = [lat, lng, intensity]
    doc = f"Point at ({lat}, {lng}), intensity: {intensity}"

    # Generate unique ID
    id = f"point_{hashlib.md5(f'{lat}-{lng}-{intensity}'.encode()).hexdigest()[:8]}"

    # Add to ChromaDB
    collection.add(
        ids=[id],
        documents=doc,
        embeddings=[vector]
    )

    # Add to JSON
    points = load_from_json()
    points.append({"lat": lat, "lng": lng, "intensity": intensity})
    save_to_json(points)

    return {"status": "success", "id": id}

# Search similar points (by vector similarity)
def find_similar(lat, lng, intensity, top_k=3):
    query_vector = [lat, lng, intensity]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "embeddings", "distances"]
    )
    docs = results["documents"]
    distances = results["distances"]

    return [
        {"doc": doc, "distance": dist}
        for doc, dist in zip(docs, distances)
    ]

# Routes
@app.route('/')
def index():
    return '''
    <h1>ChromaDB Map App (Local)</h1>
    <p>Use the HTML page at <a href="/index.html">/index.html</a> to add and search points.</p>
    <p>Points are stored in <code>data/points.json</code> and indexed in ChromaDB.</p>
    '''

@app.route('/add_point', methods=['POST'])
def add_point_route():
    data = request.json
    lat = data.get('lat')
    lng = data.get('lng')
    intensity = data.get('intensity')

    if not lat or not lng or not intensity:
        return jsonify({"error": "Missing lat, lng, or intensity"}), 400

    add_point(lat, lng, intensity)
    return jsonify({"status": "success", "message": f"Point added: ({lat}, {lng}, {intensity})"}), 200

@app.route('/search_similar', methods=['POST'])
def search_similar_route():
    data = request.json
    lat = data.get('lat')
    lng = data.get('lng')
    intensity = data.get('intensity')
    top_k = data.get('top_k', 3)

    if not lat or not lng or not intensity:
        return jsonify({"error": "Missing lat, lng, or intensity"}), 400

    results = find_similar(lat, lng, intensity, top_k)
    return jsonify({"similar_points": results})

if __name__ == '__main__':
    print("🚀 Starting ChromaDB server on http://localhost:8000")
    print("📌 Data stored at: data/points.json")
    
    # Load existing data on startup
    load_from_json()

    app.run(host='0.0.0.0', port=8000, debug=False)
