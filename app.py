from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# ---- Load data ----
# songs must include: track_name, track_id, emotions (string labels)
songs = pickle.load(open("songs.pkl", "rb"))
# similarity = pickle.load(open("similarity.pkl", "rb"))
with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)
similarity = similarity.astype(np.float16)

# Make sure 'emotions' is string for safe contains() filtering
if 'emotions' in songs.columns:
    songs['emotions'] = songs['emotions'].astype(str)
else:
    # If your column has a different name, change it here
    raise KeyError("The DataFrame must contain an 'emotions' column.")

app = Flask(__name__)

# ---- Helpers ----
def recommend(song):
    """Return top N similar tracks given a song name."""
    if song not in songs['track_name'].values:
        return []
    idx = songs.index[songs['track_name'] == song][0]
    distances = list(enumerate(similarity[idx]))
    # skip the first (itself) and take next 10
    top = sorted(distances, key=lambda x: x[1], reverse=True)[0:10]

    recs = []
    for i, _ in top:
        recs.append({
            "track_name": songs.iloc[i]['track_name'],
            "track_id":   songs.iloc[i]['track_id'],
            "emotions":   songs.iloc[i]['emotions']
        })
    return recs

def recommend_by_mood(mood_label):
    """
    Return up to 30 tracks whose 'emotions' contains the given mood label.
    We match case-insensitively and allow partial matches.
    """
    if not mood_label:
        return []
    mask = songs['emotions'].str.contains(str(mood_label), case=False, na=False)
    subset = songs.loc[mask, ['track_name', 'track_id', 'emotions']].drop_duplicates().head(15)

    return [
        {"track_name": r.track_name, "track_id": r.track_id, "emotions": r.emotions}
        for r in subset.itertuples(index=False)
    ]

# ---- Routes ----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/songs")
def get_songs():
    return jsonify({"songs": songs['track_name'].tolist()})

@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    data = request.get_json(force=True) or {}
    song_name = data.get("song", "")
    return jsonify({"recommendations": recommend(song_name)})

@app.route("/recommend_by_mood", methods=["POST"])
def recommend_by_mood_endpoint():
    data = request.get_json(force=True) or {}
    # We accept either the explicit data-name value or the visible text
    mood = (data.get("mood") or "").strip()
    return jsonify({"recommendations": recommend_by_mood(mood)})

if __name__ == "__main__":
    app.run(debug=True)
