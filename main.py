import cv2 as cv
import numpy as np
from FaceDetector import FaceDetector
from FaceEmbedder import FaceEmbedder

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

    print(f"Best score: {best_score:.3f}")

    if best_score < threshold:
        return "Unknown"
    return best_match


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
        else:
            name = "Unknown"

        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.putText(frame, name, (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()