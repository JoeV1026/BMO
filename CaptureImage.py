import cv2 as cv
from mtcnn import MTCNN

cap = cv.VideoCapture(0)
detector = MTCNN()

if not cap.isOpened():
        print("Can't open camera")
        exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't see")
        break
    
    flippedFrame = cv.flip(frame, 1)
    rgb = cv.cvtColor(flippedFrame, cv.COLOR_BGR2RGB)
    try:
        faces = detector.detect_faces(rgb)
    except Exception:
         faces = []
    if len(faces) > 0:
        for face in faces:
            if face['confidence'] > 0.9:
                x, y, w, h = face['box']
                cv.rectangle(flippedFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        

    cv.imshow("Live Camera Feed", flippedFrame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    
    