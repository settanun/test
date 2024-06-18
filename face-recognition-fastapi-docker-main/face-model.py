from PIL import Image
import face_recognition
import numpy as np
import pickle
import os

# Name and image path
known_faces = [
    ('Lisa', 'lisa.jpg'),
    ('Jennie', 'jennie.jpg'),
    ('Rose', 'rose.jpg'),
    ('Jisoo', 'jisoo.jpg'),
    ('Prime', 'Prime.jpg')
]

# Encode faces from images
known_face_names = []
known_face_encodings = []
for name, image_path in known_faces:
    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' does not exist.")
        continue

    try:
        face_image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(face_image)

        if len(face_encodings) == 0:
            print(f"Warning: No face found in '{image_path}'.")
            continue

        face_encoding = face_encodings[0]
        known_face_names.append(name)
        known_face_encodings.append(face_encoding)
    except Exception as e:
        print(f"Error processing '{image_path}': {e}")
        continue

# Dump face names and encoding to pickle
try:
    with open('faces.p', 'wb') as f:
        pickle.dump((known_face_names, known_face_encodings), f)
    print("Successfully encoded and saved faces.")
except Exception as e:
    print(f"Error saving faces to pickle: {e}")
