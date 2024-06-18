from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle
import cv2
import io
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Load face names and encoding from pickle
known_face_names, known_face_encodings = pickle.load(open('faces.p', 'rb'))

def recognize_faces_in_image(image):
    # Detect face(s) and encode them
    face_locations = face_recognition.face_locations(np.array(image))
    face_encodings = face_recognition.face_encodings(np.array(image), face_locations)

    draw = ImageDraw.Draw(image)
    face_names = []

    # Recognize face(s)
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = known_face_names[best_match_index]

        top, right, bottom, left = face_location
        draw.rectangle([left, top, right, bottom])
        draw.text((left, top), name)
        face_names.append(name)

    return face_names

@app.post("/faces_recognition/")
async def faces_recognition(image_upload: UploadFile = File(...)):
    data = await image_upload.read()
    image = Image.open(io.BytesIO(data))
    face_names = recognize_faces_in_image(image)
    return {"faces": face_names}

@app.get("/capture_and_recognize_faces/")
async def capture_and_recognize_faces():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "Failed to capture image"}

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face_names = recognize_faces_in_image(image)

    return {"faces": face_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
