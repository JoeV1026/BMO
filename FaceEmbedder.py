import torch
from facenet_pytorch import InceptionResnetV1
import cv2 as cv
import numpy as np
import pickle

class FaceEmbedder:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def preprocess(self, face):
        face = cv.resize(face, (160, 160))
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5   # 🔥 VERY IMPORTANT (normalization)
        face = np.transpose(face, (2, 0, 1))
        face = torch.tensor(face).unsqueeze(0)
        return face

    def get_embedding(self, face):
        try:
            face_tensor = self.preprocess(face)
            with torch.no_grad():
                embedding = self.model(face_tensor)
            return embedding.squeeze().numpy()
        except:
            return None

    def save_database(self, database, filepath="database.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(database, f)

    def load_database(self, filepath="database.pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)