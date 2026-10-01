import cv2
import mediapipe as mp

# MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model location
MODEL_PATH = "models/face_landmarker.task"

# Create Face Landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened.")
    exit()

print("✅ Camera started. Press Q to quit.")

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            print("❌ Could not read camera.")
            break

        # Convert OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect face landmarks
        result = landmarker.detect(mp_image)

        # Draw landmarks
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:

                height, width, _ = frame.shape

                for landmark in face_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

        cv2.imshow("Eye-Gaze - Face Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()