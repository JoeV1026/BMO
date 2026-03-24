import cv2 as cv
import numpy as np
import threading
import time

from FaceDetector import FaceDetector
from FaceEmbedder import FaceEmbedder
from speechListen import listen
from speechAPI import brain
from speechOutput import speak

current_person = "Unknown"
last_seen_time = 0

THRESHOLD = 0.65

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(embedding, database):
    bestMatch = "Unknown"
    bestScore = -1.0

    for name, embeddings in database.items():
        best = max(cosine_similarity(embedding, db_emb) for db_emb in embeddings)

        if best > bestScore:
            bestScore = best
            bestMatch = name

    if bestScore >= THRESHOLD:
        return bestMatch
    return "Unknown"

def vision_loop():
    global current_person, last_seen_time

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

        foundName = None

        for face in faces:
            x, y, w, h = face['box']
            x = max(0, x)
            y = max(0, y)

            cropped = frame[y:y+h, x:x+w]
            if cropped.size == 0:
                continue

            embedding = embedder.get_embedding(cropped)

            if embedding is not None:
                name = recognize(embedding, database)
                if name != "Unknown":
                    foundName = name
            else:
                name = "Unknown"

            cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv.putText(frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if foundName is not None:
            current_person = foundName
            last_seen_time = time.time()
        elif time.time() - last_seen_time > 3.0:
            current_person = "Unknown"

        cv.imshow("Face Recognition", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def speech_loop():
    global current_person

    while True:
        text = listen()
        if not text:
            continue

        userText = text.strip().lower()
        print("You said:", userText)

        if "who am i" in userText or "what is my name" in userText:
            if current_person != "Unknown":
                response = f"You are {current_person}."
            else:
                response = "I don't know yet."
        else:
            response = brain(userText)

        print("Robot:", response)
        speak(response)

        time.sleep(0.3)

def main():
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()

    speech_loop()

if __name__ == "__main__":
    main()