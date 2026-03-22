import os
import cv2 as cv
from FaceEmbedder import FaceEmbedder

dataset_path = "dataset"

embedder = FaceEmbedder()
database = {}

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    database[person] = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv.imread(img_path)
        if img is None:
            continue

        embedding = embedder.get_embedding(img)

        if embedding is not None:
            database[person].append(embedding)

embedder.save_database(database)

print("Database built successfully!")