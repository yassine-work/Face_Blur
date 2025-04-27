from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
import numpy as np
import tempfile
import os
import io  # Required to handle in-memory video files
from main import FaceMeshDetector, blur_face_shape

app = FastAPI()


@app.post("/uploadimage/")
async def upload_image(file: UploadFile = File(...)):
    img_data = await file.read()
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    img = process_image(img)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, img)

    response = FileResponse(temp_file.name, media_type="image/jpeg")

    os.remove(temp_file.name)

    return response


@app.post("/uploadvideo/")
async def upload_video(file: UploadFile = File(...)):
    video_data = await file.read()
    video_stream = io.BytesIO(video_data)

    processed_video_path = process_video(video_stream)

    response = FileResponse(processed_video_path, media_type="video/mp4")

    os.remove(processed_video_path)

    return response


def process_image(img):
    detector = FaceMeshDetector()

    faces = detector.findFaceContours(img)

    for face in faces:
        img = blur_face_shape(img, face)

    return img


def process_video(video_stream):
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_video_file.name, 'wb') as f:
        f.write(video_stream.read())

    cap = cv2.VideoCapture(temp_video_file.name)

    if not cap.isOpened():
        print("Error opening video stream or file")
        os.remove(temp_video_file.name)  # Clean up temporary video file
        return None

    output_path = "processed_video.mp4"


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 video codec
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    detector = FaceMeshDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        faces = detector.findFaceContours(frame)

        for face_outline in faces:
            frame = blur_face_shape(frame, face_outline)

        out.write(frame)

    cap.release()
    out.release()

    os.remove(temp_video_file.name)

    return output_path
