import cv2
import mediapipe as mp
import pyautogui

# --------------------------------
# Screen size
# --------------------------------

screen_width, screen_height = pyautogui.size()

# --------------------------------
# MediaPipe setup
# --------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# --------------------------------
# Camera
# --------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

print()
print("================================")
print(" SMART EYE-GAZE MOUSE")
print("================================")
print("Move ONLY your eyes.")
print("Press Q to quit.")
print("================================")

# --------------------------------
# Cursor settings
# --------------------------------

current_x = screen_width // 2
current_y = screen_height // 2

# Higher = faster cursor
smoothing = 0.35

# Eye movement amplification
sensitivity_x = 2.2
sensitivity_y = 2.2

# --------------------------------
# Start MediaPipe
# --------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # --------------------------------
        # Convert image
        # --------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        if result.face_landmarks:

            face = result.face_landmarks[0]

            # --------------------------------
            # Iris
            # --------------------------------

            left_iris = face[468]
            right_iris = face[473]

            eye_x = (
                left_iris.x + right_iris.x
            ) / 2

            eye_y = (
                left_iris.y + right_iris.y
            ) / 2

            # --------------------------------
            # Amplify eye movement
            # --------------------------------

            center_x = 0.5
            center_y = 0.5

            relative_x = (
                eye_x - center_x
            ) * sensitivity_x

            relative_y = (
                eye_y - center_y
            ) * sensitivity_y

            # Keep inside screen
            relative_x = max(
                -0.5,
                min(0.5, relative_x)
            )

            relative_y = max(
                -0.5,
                min(0.5, relative_y)
            )

            target_x = int(
                (relative_x + 0.5)
                * screen_width
            )

            target_y = int(
                (relative_y + 0.5)
                * screen_height
            )

            # --------------------------------
            # Smooth cursor
            # --------------------------------

            current_x += (
                target_x - current_x
            ) * smoothing

            current_y += (
                target_y - current_y
            ) * smoothing

            # --------------------------------
            # Move cursor
            # --------------------------------

            pyautogui.moveTo(
                int(current_x),
                int(current_y),
                duration=0
            )

            # --------------------------------
            # Draw iris
            # --------------------------------

            left_x = int(
                left_iris.x * width
            )

            left_y = int(
                left_iris.y * height
            )

            right_x = int(
                right_iris.x * width
            )

            right_y = int(
                right_iris.y * height
            )

            cv2.circle(
                frame,
                (left_x, left_y),
                6,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                (right_x, right_y),
                6,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                "EYE MOUSE ACTIVE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                "Move ONLY your eyes",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # --------------------------------
        # Camera window
        # --------------------------------

        cv2.imshow(
            "Smart Eye-Gaze Mouse",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

print("Eye mouse closed.")