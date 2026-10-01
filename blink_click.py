import cv2
import mediapipe as mp
import pyautogui
import time

print("Starting Smart Eye-Gaze Mouse + Blink Click...")

screen_width, screen_height = pyautogui.size()

# ============================================================
# MEDIAPIPE
# ============================================================

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

# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

print("Move ONLY your eyes.")
print("Blink = LEFT CLICK")
print("Press Q = EXIT")

# ============================================================
# CURSOR SETTINGS
# ============================================================

current_x = screen_width // 2
current_y = screen_height // 2

# Lower movement = steadier cursor
smoothing = 0.18

# Eye movement sensitivity
sensitivity_x = 3.5
sensitivity_y = 3.5

# Ignore tiny eye movements
dead_zone = 0.07

# ============================================================
# BLINK SETTINGS
# ============================================================

blink_threshold = 0.20
blink_start = None
last_click = 0
click_cooldown = 1.0


# ============================================================
# EYE RATIO
# ============================================================

def eye_ratio(face, top, bottom, left, right):

    vertical_distance = abs(
        face[top].y - face[bottom].y
    )

    horizontal_distance = abs(
        face[left].x - face[right].x
    )

    if horizontal_distance < 0.001:
        return 1.0

    return vertical_distance / horizontal_distance


# ============================================================
# RELATIVE EYE POSITION
# ============================================================

def horizontal_ratio(iris, outer, inner):

    width = abs(inner.x - outer.x)

    if width < 0.001:
        return 0.5

    value = (
        iris.x -
        min(outer.x, inner.x)
    ) / width

    return max(0.0, min(1.0, value))


def vertical_ratio(iris, top, bottom):

    height = abs(bottom.y - top.y)

    if height < 0.001:
        return 0.5

    value = (
        iris.y -
        min(top.y, bottom.y)
    ) / height

    return max(0.0, min(1.0, value))


# ============================================================
# START
# ============================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        status = "FACE NOT DETECTED"

        if result.face_landmarks:

            face = result.face_landmarks[0]

            # =================================================
            # LANDMARKS
            # =================================================

            left_iris = face[468]
            right_iris = face[473]

            left_outer = face[33]
            left_inner = face[133]

            right_outer = face[362]
            right_inner = face[263]

            left_top = face[159]
            left_bottom = face[145]

            right_top = face[386]
            right_bottom = face[374]

            # =================================================
            # EYE POSITION
            # =================================================

            left_h = horizontal_ratio(
                left_iris,
                left_outer,
                left_inner
            )

            right_h = horizontal_ratio(
                right_iris,
                right_outer,
                right_inner
            )

            eye_h = (
                left_h + right_h
            ) / 2

            left_v = vertical_ratio(
                left_iris,
                left_top,
                left_bottom
            )

            right_v = vertical_ratio(
                right_iris,
                right_top,
                right_bottom
            )

            eye_v = (
                left_v + right_v
            ) / 2

            # =================================================
            # CENTER
            # =================================================

            movement_x = eye_h - 0.50
            movement_y = eye_v - 0.50

            # =================================================
            # DEAD ZONE
            # =================================================

            if abs(movement_x) < dead_zone:
                movement_x = 0

            if abs(movement_y) < dead_zone:
                movement_y = 0

            # =================================================
            # SENSITIVITY
            # =================================================

            movement_x *= sensitivity_x
            movement_y *= sensitivity_y

            movement_x = max(
                -0.5,
                min(0.5, movement_x)
            )

            movement_y = max(
                -0.5,
                min(0.5, movement_y)
            )

            # =================================================
            # TARGET
            # =================================================

            target_x = int(
                (movement_x + 0.5)
                * screen_width
            )

            target_y = int(
                (movement_y + 0.5)
                * screen_height
            )

            # =================================================
            # STRONG SMOOTHING
            # =================================================

            current_x = (
                current_x * (1 - smoothing)
                + target_x * smoothing
            )

            current_y = (
                current_y * (1 - smoothing)
                + target_y * smoothing
            )

            # =================================================
            # MOVE MOUSE
            # =================================================

            pyautogui.moveTo(
                int(current_x),
                int(current_y),
                duration=0
            )

            # =================================================
            # BLINK DETECTION
            # =================================================

            left_eye = eye_ratio(
                face,
                159,
                145,
                33,
                133
            )

            right_eye = eye_ratio(
                face,
                386,
                374,
                362,
                263
            )

            average_eye = (
                left_eye + right_eye
            ) / 2

            if average_eye < blink_threshold:

                status = "BLINK DETECTED"

                if blink_start is None:
                    blink_start = time.time()

                duration = (
                    time.time() -
                    blink_start
                )

                if (
                    duration >= 0.08
                    and
                    time.time() - last_click
                    >= click_cooldown
                ):

                    pyautogui.click()

                    print("LEFT CLICK")

                    last_click = time.time()

            else:

                blink_start = None
                status = "EYES OPEN"

            # =================================================
            # DRAW IRIS
            # =================================================

            lx = int(left_iris.x * width)
            ly = int(left_iris.y * height)

            rx = int(right_iris.x * width)
            ry = int(right_iris.y * height)

            cv2.circle(
                frame,
                (lx, ly),
                6,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                (rx, ry),
                6,
                (0, 255, 0),
                -1
            )

            # =================================================
            # TEXT
            # =================================================

            cv2.putText(
                frame,
                status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                "MOVE EYES - BLINK TO CLICK",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # =====================================================
        # CAMERA WINDOW
        # =====================================================

        cv2.imshow(
            "Smart Eye-Gaze Control",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()

print("Smart Eye-Gaze system closed.")