import cv2 as cv
from FaceDetector import FaceDetector

cap = cv.VideoCapture(0)
detector = FaceDetector()

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()

    frame = cv.flip(frame, 1)

    faces = detector.detectFaces(frame)
    frame = detector.drawFaces(frame, faces)

    cv.imshow("Face Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()