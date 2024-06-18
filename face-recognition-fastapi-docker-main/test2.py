from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle
import cv2
import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
import geopy
from geopy.geocoders import Nominatim
import folium
import datetime

app = FastAPI()

# Enable CORS
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

def get_current_location():
    # Here you should implement a method to get the current location
    # For example, if you're deploying this on a mobile device,
    # you can use the device's GPS sensor to get the current location
    # This function should return the latitude and longitude
    # For demonstration purposes, I'll just return some sample coordinates
    return 13.7563, 100.5018  # Bangkok, Thailand

def create_map(latitude, longitude):
    map = folium.Map(location=[latitude, longitude], zoom_start=15)
    folium.Marker(location=[latitude, longitude], popup="You're here", icon=folium.Icon(color="blue")).add_to(map)
    return map

def log_result(log_path, faces, latitude, longitude):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Recognized Faces: {', '.join(faces)}\n")
        f.write(f"Latitude: {latitude}, Longitude: {longitude}\n\n")

@app.post("/faces_recognition/")
async def faces_recognition(image_upload: UploadFile = File(...)):
    data = await image_upload.read()
    image = Image.open(io.BytesIO(data))
    face_names = recognize_faces_in_image(image)

    # Log result
    log_result("C:/Users/4PLUS/test/face-recognition-fastapi-docker-main/log/log.txt", face_names, None, None)

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

    # Log result
    log_result("C:/Users/4PLUS/test/face-recognition-fastapi-docker-main/log/log.txt", face_names, None, None)

    return {"faces": face_names}

@app.get("/current_location/")
async def current_location():
    latitude, longitude = get_current_location()

    # Log result
    log_result("C:/Users/4PLUS/test/face-recognition-fastapi-docker-main/log/log.txt", [], latitude, longitude)

    return {"latitude": latitude, "longitude": longitude}

@app.get("/map/")
async def show_map():
    latitude, longitude = get_current_location()
    map = create_map(latitude, longitude)
    return map._repr_html_()

@app.get("/")
async def get_index():
    return HTMLResponse("""
    <html>
    <head>
        <title>Face Recognition App</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto py-8 text-center">
            <h1 class="text-3xl font-bold mb-4">Face Recognition App</h1>
            <button id="actionBtn" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded">Perform Action</button>
            <div id="result" class="mt-4"></div>
        </div>
        <script>
            const actionBtn = document.getElementById('actionBtn');
            const resultDiv = document.getElementById('result');

            actionBtn.addEventListener('click', async () => {
                const response1 = await fetch('/capture_and_recognize_faces/');
                const data1 = await response1.json();
                
                const response2 = await fetch('/current_location/');
                const data2 = await response2.json();

                const response3 = await fetch('/map/');
                const html = await response3.text();

                resultDiv.innerHTML = `<p class="text-xl font-semibold">Recognized Faces: ${data1.faces.join(', ')}</p>
                                       <p class="text-xl font-semibold">Latitude: ${data2.latitude}, Longitude: ${data2.longitude}</p>
                                       ${html}`;
            });
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
