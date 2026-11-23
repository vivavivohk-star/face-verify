import numpy as np
import face_recognition

CHALLENGES = ("turn_left", "turn_right")

INSTRUCTIONS = {
    "turn_left": "Slowly turn your head to your left, then back to center.",
    "turn_right": "Slowly turn your head to your right, then back to center.",
}

MIN_VALID_FRAMES = 6
YAW_MOVE = 0.20


def analyze_frame(rgb):
    locations = face_recognition.face_locations(rgb, model="hog")
    if len(locations) != 1:
        return None
    landmarks = face_recognition.face_landmarks(rgb, locations, model="large")
    encodings = face_recognition.face_encodings(rgb, locations, num_jitters=1)
    if not landmarks or not encodings:
        return None
    lm = landmarks[0]
    left_eye = np.array(lm["left_eye"], dtype=float).mean(axis=0)
    right_eye = np.array(lm["right_eye"], dtype=float).mean(axis=0)
    eye_mid = (left_eye + right_eye) / 2.0
    eye_span = float(np.linalg.norm(right_eye - left_eye)) or 1.0
    nose = np.array(lm["nose_bridge"][-1], dtype=float)
    metrics = {
        "yaw": float((nose[0] - eye_mid[0]) / eye_span),
    }
    return metrics, np.asarray(encodings[0], dtype=np.float64)


def motion_satisfied(challenge, samples):
    yaw = np.array([s["yaw"] for s in samples])
    if challenge == "turn_left":
        return float(yaw.max() - np.median(yaw)) >= YAW_MOVE
    if challenge == "turn_right":
        return float(np.median(yaw) - yaw.min()) >= YAW_MOVE
    return False