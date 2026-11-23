import base64
import binascii
import io
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps

import face_recognition
import numpy as np
from flask import Flask, jsonify, render_template, request, session
from PIL import Image, UnidentifiedImageError

from liveness import INSTRUCTIONS, CHALLENGES, MIN_VALID_FRAMES, analyze_frame, motion_satisfied

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("FV_DB_PATH", os.path.join(BASE_DIR, "face_verify.db"))
MATCH_THRESHOLD = float(os.environ.get("FV_MATCH_THRESHOLD", "0.42"))
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z '\-]{1,39}$")
CHALLENGE_SECONDS = 4
FRAME_COUNT = 10
MAX_FRAMES = 12
CHALLENGE_WINDOW_SECONDS = 25

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FV_SECRET_KEY") or secrets.token_hex(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES * 3


@contextmanager
def db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_key TEXT UNIQUE NOT NULL,
                face_encoding BLOB NOT NULL,
                consented_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT
            )
            """
        )


def utc_now():
    return datetime.now(timezone.utc)


def parse_time(value):
    return datetime.fromisoformat(value) if value else None


def normalize_name(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def decode_image(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        return None
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        return None
    if image.width * image.height > 4000 * 4000:
        return None
    return np.array(image.convert("RGB"))


def extract_single_face(rgb):
    locations = face_recognition.face_locations(rgb, model="hog")
    if len(locations) != 1:
        return None, len(locations)
    encodings = face_recognition.face_encodings(rgb, locations, num_jitters=2)
    if not encodings:
        return None, 0
    return np.asarray(encodings[0], dtype=np.float64), 1


def encode_for_storage(vector):
    return np.asarray(vector, dtype=np.float64).tobytes()


def decode_from_storage(blob):
    return np.frombuffer(blob, dtype=np.float64)


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"ok": False, "error": "Not signed in."}), 401
        return view(*args, **kwargs)

    return wrapper


def read_json():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/create")
def create():
    data = read_json()
    name = normalize_name(data.get("name"))

    if data.get("consent") is not True:
        return jsonify({"ok": False, "error": "Please confirm consent to continue."}), 400
    if not NAME_PATTERN.match(name):
        return jsonify({"ok": False, "error": "Use 2-40 letters, spaces, hyphens, or apostrophes."}), 400

    rgb = decode_image(data.get("image"))
    if rgb is None:
        return jsonify({"ok": False, "error": "Could not read the photo. Capture it again."}), 400

    vector, count = extract_single_face(rgb)
    if vector is None:
        message = (
            "No face detected. Face the camera in good light."
            if count == 0
            else "More than one face detected. Only one person should be in the frame."
        )
        return jsonify({"ok": False, "error": message}), 400

    now = utc_now().isoformat()
    try:
        with db_connection() as conn:
            conn.execute(
                """
                INSERT INTO people (name, name_key, face_encoding, consented_at, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, name.lower(), encode_for_storage(vector), now, now),
            )
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "A profile with that name already exists."}), 409

    return jsonify({"ok": True, "name": name})


@app.get("/api/challenge")
def challenge():
    name = secrets.choice(CHALLENGES)
    session["challenge"] = {"name": name, "issued": utc_now().isoformat()}
    return jsonify({
        "ok": True,
        "challenge": name,
        "instruction": INSTRUCTIONS[name],
        "seconds": CHALLENGE_SECONDS,
        "frames": FRAME_COUNT,
    })


@app.post("/api/verify")
def verify():
    data = read_json()
    frames = data.get("frames")
    pending = session.pop("challenge", None)
    failure = "We could not match that scan. Try again."

    if not isinstance(frames, list) or not (MIN_VALID_FRAMES <= len(frames) <= MAX_FRAMES):
        return jsonify({"ok": False, "error": "The scan was incomplete. Start again."}), 400
    if not pending:
        return jsonify({"ok": False, "error": "Start a new scan."}), 400
    issued = parse_time(pending["issued"])
    if utc_now() - issued > timedelta(seconds=CHALLENGE_WINDOW_SECONDS):
        return jsonify({"ok": False, "error": "The scan expired. Start again."}), 400

    samples, encodings = [], []
    try:
        for frame in frames:
            rgb = decode_image(frame)
            if rgb is None:
                continue
            result = analyze_frame(rgb)
            if result is None:
                continue
            metrics, encoding = result
            samples.append(metrics)
            encodings.append(encoding)
    except Exception:
        app.logger.exception("Face analysis failed")
        return jsonify({"ok": False, "error": "The face analysis failed on the server. Check the console for details."}), 500

    face_ok = False
    error = failure
    matched_row = None
    now = utc_now()
    if len(samples) < MIN_VALID_FRAMES:
        error = "Could not get a clear view of exactly one face. Face the camera in good light."
    elif not motion_satisfied(pending["name"], samples):
        error = "Movement not detected. " + INSTRUCTIONS[pending["name"]]
    else:
        with db_connection() as conn:
            rows = conn.execute("SELECT * FROM people").fetchall()
        best_row = None
        best_dist = None
        for row in rows:
            stored = decode_from_storage(row["face_encoding"])
            distances = face_recognition.face_distance(encodings, stored)
            d = float(np.min(distances))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_row = row
        if best_row is not None:
            locked_until = parse_time(best_row["locked_until"])
            if locked_until and locked_until > now:
                return jsonify({"ok": False, "error": "Too many failed attempts. Try again later."}), 429
            if best_dist is not None and best_dist <= MATCH_THRESHOLD:
                face_ok = True
                matched_row = best_row
            else:
                error = failure
                matched_row = best_row

    if face_ok and matched_row is not None:
        with db_connection() as conn:
            conn.execute(
                "UPDATE people SET failed_attempts = 0, locked_until = NULL WHERE id = ?",
                (matched_row["id"],),
            )
        session.clear()
        session["user_id"] = matched_row["id"]
        session["name"] = matched_row["name"]
        return jsonify({"ok": True, "name": matched_row["name"]})

    if matched_row is not None:
        attempts = matched_row["failed_attempts"] + 1
        new_lock = None
        if attempts >= MAX_FAILED_ATTEMPTS:
            new_lock = (now + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
            attempts = 0
        with db_connection() as conn:
            conn.execute(
                "UPDATE people SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                (attempts, new_lock, matched_row["id"]),
            )
    return jsonify({"ok": False, "error": error}), 401


@app.get("/api/me")
@login_required
def me():
    return jsonify({"ok": True, "name": session.get("name")})


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.post("/api/delete")
@login_required
def delete_profile():
    with db_connection() as conn:
        conn.execute("DELETE FROM people WHERE id = ?", (session["user_id"],))
    session.clear()
    return jsonify({"ok": True, "message": "Your profile and face template were deleted."})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
else:
    init_db()
