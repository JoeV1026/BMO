import cv2 as cv
import os
from FaceDetector import FaceDetector

person = "Test"
angles = ["center", "left", "right", "up", "down"]
numImgs = 100

baseDir = f"dataset/{person}"
os.makedirs(baseDir, exist_ok=True)

for a in angles:
    os.makedirs(f"{baseDir}/{a}", exist_ok=True)

cap = cv.VideoCapture(0)
detector = FaceDetector()

currentAngle = 0
count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    faces = detector.detect_faces(frame)

    instruction = f"Look {angles[currentAngle]}"
    cv.putText(frame, instruction, (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if len(faces) > 0:
        x, y, w, h = faces[0]["box"]

        face = frame[y:y+h, x:x+w]

        if face.size != 0:
            face = cv.resize(face, (160,160))

            savePath = f"{baseDir}/{angles[currentAngle]}/{count:04d}.jpg"
            cv.imwrite(savePath, face)
            count += 1

        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    if count >= numImgs:
        count = 0
        currentAngle += 1

        if currentAngle >= len(angles):
            print("complete")
            break

    cv.imshow("Dataset Collector", frame)

    if cv.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()