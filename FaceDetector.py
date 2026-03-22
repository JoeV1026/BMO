import cv2 as cv
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
    
    def detectFaces(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        try:
            faces = self.detector.detect_faces(rgb)
        except Exception:
            faces = []
        return faces
    def drawFaces(self, frame, faces):
        for face in faces:
            x, y, w, h = face['box']
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame