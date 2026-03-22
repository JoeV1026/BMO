import insightface
import pickle

class FaceEmbedder:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis()
        self.app.prepare(ctx_id=0)

    def get_embedding(self, frame):
        faces = self.app.get(frame)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    def save_database(self, database, filepath="database.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(database, f)

    def load_database(self, filepath="database.pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)