import cv2
import mediapipe as mp
import time
import numpy as np


# face contour
FACE_CONTOUR_IDXS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )

    def findFaceContours(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                ih, iw, _ = img.shape
                face = []
                for idx in FACE_CONTOUR_IDXS:
                    lm = faceLms.landmark[idx]
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return faces


def blur_face_shape(img, face_outline):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    points = np.array(face_outline, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)

    mask = cv2.GaussianBlur(mask, (21, 21), 11)

    blurred = cv2.GaussianBlur(img, (51, 51), 30)

    mask_normalized = mask[:, :, np.newaxis] / 255.0
    output = (img * (1 - mask_normalized) + blurred * mask_normalized).astype(np.uint8)

    return output


def main():
    cap = cv2.VideoCapture(1)
    detector = FaceMeshDetector()
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        faces = detector.findFaceContours(img)

        for face_outline in faces:
            img = blur_face_shape(img, face_outline)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 78), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

        cv2.imshow("Smooth Face Blur", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main1(path):   #photo
    img_path = path
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found or failed to load.")
        return

    # Resize the image for better processing if it's too large
    height, width = img.shape[:2]
    new_width = 800
    aspect_ratio = new_width / width
    new_height = int(height * aspect_ratio)
    img = cv2.resize(img, (new_width, new_height))

    detector = FaceMeshDetector()
    faces = detector.findFaceContours(img)

    for face_outline in faces:
        # Apply blur to face shape
        img = blur_face_shape(img, face_outline)

    cv2.imshow("Smooth Face Blur", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
    #main1(.....)