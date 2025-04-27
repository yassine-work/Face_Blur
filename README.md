# Face Blur Application

This project uses OpenCV and MediaPipe to apply a blur effect to faces in images and videos. The application includes both a **FastAPI-based backend** for image/video processing and a **simple web interface** for uploading and receiving the processed media. It has been **dockerized** for easy deployment. The project is hosted and tested on **Render**.

## Demo
You can watch the demo of the application [here](https://drive.google.com/file/d/1d3MZ1u-ZAOljvY6OeFmF4MDu3cEPIHel/view?usp=drive_link).

## Features
- **Image Upload & Blur:** Upload an image, and all faces will be blurred.
- **Video Upload & Blur:** Upload a video, and the faces in each frame will be blurred.
- **Web Interface:** A simple HTML form for uploading images or videos.
- **FastAPI Backend:** The backend processes images and videos to blur faces.
- **Docker Support:** The app is dockerized for easy deployment.
- **Deploy on Render:** The app is deployed on Render, providing a live testing environment.

## Code Explanation

### Face Mesh Detection
The application uses **MediaPipe** to detect faces in images or videos by locating facial landmarks. **OpenCV** is used to process images and apply the blur effect. Here's how it works:

1. **Face Contours:**
   The script defines a list of face contour indices (`FACE_CONTOUR_IDXS`) that correspond to specific points on the face (like the edges of the face, around the eyes, nose, and chin). These are used to create a mask for the blur effect.

2. **FaceMeshDetector Class:**
   - The `FaceMeshDetector` class is responsible for detecting faces using MediaPipe. It uses a pre-trained **FaceMesh** model to identify facial landmarks.
   - The `findFaceContours` method takes an image, processes it, and extracts the coordinates of the facial landmarks. These landmarks represent the outline of the face, which will later be blurred.

### Blurring Faces
Once the face contours are detected, the application applies a blur effect to each face.

3. **Blurring Logic:**
   - The `blur_face_shape` function creates a mask from the detected face contours. This mask is blurred using a Gaussian blur.
   - The input image is then also blurred using a stronger Gaussian blur.
   - The blurred image and the original image are combined, where the face area is replaced by the blurred version, and the rest of the image remains unchanged.

### Main Functions

- **main()**: This function is responsible for processing a video stream (from a webcam or a file) in real time. It:
  - Captures video frames.
  - Detects faces in each frame.
  - Applies the blur effect to each face.
  - Displays the processed video with FPS displayed.
  
- **main1(path)**: This function processes a single image. It:
  - Reads the image from the given file path.
  - Detects faces in the image.
  - Applies the blur effect to the faces.
  - Displays the processed image.

### FastAPI Backend
- **API Endpoints:**
  - **`/uploadimage/`**: Uploads an image, processes it, and returns the image with blurred faces.
  - **`/uploadvideo/`**: Uploads a video, processes it frame-by-frame, and returns the video with blurred faces.
  - **`/healthz`**: A health check endpoint to verify if the service is running properly.

### Deployment on Render
The application is deployed on **Render**, a platform for hosting web applications, making it easier to test and access the API and web interface. The Dockerized app is automatically deployed and made accessible via a public URL.

## Installation

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-blur-project.git
   cd face-blur-project
2.Install dependencies:
pip install -r requirements.txt

3.Run the FastAPI application:
uvicorn api_application:app --reload

4.Access the app at http://127.0.0.1:8000.

Running with Docker
1.Build the Docker image:
docker build -t face-blur-app .
2.Run the Docker container:
docker run -d -p 80:80 face-blur-app
3.Access the app at http://localhost.

