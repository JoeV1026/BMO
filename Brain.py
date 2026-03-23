import time
import threading
import cv2 as cv
import numpy as np

from FaceDetector import FaceDetector
from FaceEmbedder import FaceEmbedder
from speechListen import listen
from speechAPI import brain
from speechOutput import speak


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize(embedding, database, threshold=0.72, min_embeddings_per_person=3):
    best_match = "Unknown"
    best_score = -1.0

    for name, embeddings in database.items():
        if len(embeddings) < min_embeddings_per_person:
            continue

        person_best = max(cosine_similarity(embedding, db_emb) for db_emb in embeddings)

        if person_best > best_score:
            best_score = person_best
            best_match = name

    return best_match if best_score >= threshold else "Unknown"


state_lock = threading.Lock()
current_person = "Unknown"
last_seen_time = 0.0

last_greeted_name = None
last_greeted_time = 0.0
GREET_COOLDOWN_SECONDS = 20


def vision_loop():
    global current_person, last_seen_time, last_greeted_name, last_greeted_time

    cap = cv.VideoCapture(0)
    detector = FaceDetector()
    embedder = FaceEmbedder()
    database = embedder.load_database()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        faces = detector.detectFaces(frame)

        seen_name = "Unknown"
        best_face_area = 0

        for face in faces:
            x, y, w, h = face["box"]

            x = max(0, x)
            y = max(0, y)
            cropped = frame[y:y + h, x:x + w]

            if cropped.size == 0:
                continue

            embedding = embedder.get_embedding(cropped)
            name = recognize(embedding, database) if embedding is not None else "Unknown"

            area = w * h
            if area > best_face_area:
                best_face_area = area
                seen_name = name

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        now = time.time()

        with state_lock:
            current_person = seen_name
            if seen_name != "Unknown":
                last_seen_time = now

                should_greet = (
                    seen_name != last_greeted_name
                    or (now - last_greeted_time) > GREET_COOLDOWN_SECONDS
                )

                if should_greet:
                    last_greeted_name = seen_name
                    last_greeted_time = now
                    greet_text = f"Hello, {seen_name}."
                else:
                    greet_text = None
            else:
                greet_text = None

        if greet_text:
            speak(greet_text)

        cv.imshow("Brain Vision", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def speech_loop():
    global current_person, last_seen_time

    while True:
        text = listen()
        if not text:
            continue

        user_text = text.strip().lower()
        print("You said:", user_text)

        if "who am i" in user_text or "what is my name" in user_text:
            with state_lock:
                if time.time() - last_seen_time <= 3.0 and current_person != "Unknown":
                    response = f"You are {current_person}."
                else:
                    response = "I don't know yet. Please look at the camera."
        else:
            response = brain(user_text)

        print("Robot:", response)
        speak(response)


def main():
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()
    speech_loop()


if __name__ == "__main__":
    main()
