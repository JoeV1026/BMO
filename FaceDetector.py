import cv2 as cv
from facenet_pytorch import MTCNN

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN(keep_all=True)

    def detectFaces(self, frame):
        # Convert BGR → RGB (IMPORTANT FIX)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        boxes, _ = self.detector.detect(rgb)

        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                faces.append({"box": (x, y, w, h)})

        return faces

    def drawFaces(self, frame, faces):
        for face in faces:
            x, y, w, h = face['box']
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame