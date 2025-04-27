from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import io
import cv2
import numpy as np
from main import FaceMeshDetector, blur_face_shape

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def remove_file(file_path: str):
    try:
        os.unlink(file_path)
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        with open("index.html", "r") as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load HTML: {str(e)}")

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

@app.post("/uploadimage/")
async def upload_image(file: UploadFile = File(...)):
    try:
        img_data = await file.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        img = process_image(img)

        success, encoded_img = cv2.imencode('.jpg', img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image")

        return Response(
            content=encoded_img.tobytes(),
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=processed.jpg"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/uploadvideo/")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        video_data = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_path = temp_file.name
            temp_file.write(video_data)

        processed_video_path = process_video(temp_path)
        background_tasks.add_task(remove_file, temp_path)

        response = FileResponse(
            processed_video_path,
            media_type="video/mp4",
            filename="processed_video.mp4"
        )
        background_tasks.add_task(remove_file, processed_video_path)

        return response
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            background_tasks.add_task(remove_file, temp_path)
        if 'processed_video_path' in locals() and os.path.exists(processed_video_path):
            background_tasks.add_task(remove_file, processed_video_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

def process_image(img):
    try:
        detector = FaceMeshDetector()
        faces = detector.findFaceContours(img)

        for face in faces:
            img = blur_face_shape(img, face)

        return img
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

def process_video(input_path: str) -> str:
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Use temp directory for output
        output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 20.0, (width, height))

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

        return output_path
    except Exception as e:
        if 'out' in locals():
            out.release()
        if 'cap' in locals():
            cap.release()
        raise