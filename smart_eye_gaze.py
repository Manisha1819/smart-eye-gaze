
import cv2
import mediapipe as mp
import pyautogui
import time
import json

# ============================================================
# SMART EYE-GAZE
# Eye Cursor + Scrolling + Blink Click
# ============================================================

print()
print("==============================================")
print("        SMART EYE-GAZE SYSTEM")
print("==============================================")
print("Starting...")
print()

# ============================================================
# SCREEN
# ============================================================

screen_width, screen_height = pyautogui.size()

# ============================================================
# LOAD CALIBRATION
# ============================================================

try:
    with open("calibration.json", "r") as file:
        calibration = json.load(file)

    CENTER_H = calibration["CENTER"]["h"]
    CENTER_V = calibration["CENTER"]["v"]

    print("Calibration loaded successfully.")

except Exception:
    CENTER_H = 0.50
    CENTER_V = 0.50

    print("Calibration file not found.")
    print("Using default calibration.")

print(f"Center H = {CENTER_H:.3f}")
print(f"Center V = {CENTER_V:.3f}")

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
    print("ERROR: Camera could not be opened.")
    exit()

print()
print("Camera opened successfully.")
print()
print("CONTROLS")
print("----------------------------------------------")
print("Move eyes       → Move cursor")
print("Look TOP        → Scroll UP")
print("Look BOTTOM     → Scroll DOWN")
print("Blink           → Left Click")
print("Press Q         → Exit")
print("----------------------------------------------")
print()

# ============================================================
# CURSOR SETTINGS
# ============================================================

current_x = screen_width // 2
current_y = screen_height // 2

smooth_factor = 0.12

horizontal_sensitivity = 7.0
vertical_sensitivity = 8.0

dead_zone_x = 0.025
dead_zone_y = 0.025

# ============================================================
# FILTER
# ============================================================

filtered_h = CENTER_H
filtered_v = CENTER_V

filter_strength = 0.20

# ============================================================
# SCROLL
# ============================================================

scroll_zone_top = 0.18
scroll_zone_bottom = 0.82

scroll_amount = 35

last_scroll = 0
scroll_delay = 0.02

# ============================================================
# BLINK
# ============================================================

blink_threshold = 0.20

blink_start = None
last_click = 0

# IMPORTANT:
# Prevents one long blink from producing many clicks.
blink_clicked = False

click_cooldown = 1.0


# ============================================================
# EYE RATIO
# ============================================================

def eye_ratio(face, top, bottom, left, right):

    vertical = abs(
        face[top].y -
        face[bottom].y
    )

    horizontal = abs(
        face[left].x -
        face[right].x
    )

    if horizontal < 0.001:
        return 1.0

    return vertical / horizontal


# ============================================================
# HORIZONTAL EYE POSITION
# ============================================================

def horizontal_ratio(iris, outer, inner):

    width = abs(
        inner.x -
        outer.x
    )

    if width < 0.001:
        return 0.5

    value = (
        iris.x -
        min(outer.x, inner.x)
    ) / width

    return max(
        0.0,
        min(1.0, value)
    )


# ============================================================
# VERTICAL EYE POSITION
# ============================================================

def vertical_ratio(iris, top, bottom):

    height = abs(
        bottom.y -
        top.y
    )

    if height < 0.001:
        return 0.5

    value = (
        iris.y -
        min(top.y, bottom.y)
    ) / height

    return max(
        0.0,
        min(1.0, value)
    )


# ============================================================
# MAIN MEDIAPIPE LOOP
# ============================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        # ====================================================
        # MIRROR CAMERA
        # ====================================================

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # ====================================================
        # CONVERT IMAGE
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        status = "NO FACE"

        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.face_landmarks:

            face = result.face_landmarks[0]

            # =================================================
            # IRIS LANDMARKS
            # =================================================

            left_iris = face[468]
            right_iris = face[473]

            # =================================================
            # EYE BOUNDARIES
            # =================================================

            left_outer = face[33]
            left_inner = face[133]

            right_outer = face[362]
            right_inner = face[263]

            left_top = face[159]
            left_bottom = face[145]

            right_top = face[386]
            right_bottom = face[374]

            # =================================================
            # HORIZONTAL POSITION
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

            raw_h = (
                left_h +
                right_h
            ) / 2

            # =================================================
            # VERTICAL POSITION
            # =================================================

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

            raw_v = (
                left_v +
                right_v
            ) / 2

            # =================================================
            # FILTER
            # =================================================

            filtered_h = (
                filtered_h *
                (1 - filter_strength)
                +
                raw_h *
                filter_strength
            )

            filtered_v = (
                filtered_v *
                (1 - filter_strength)
                +
                raw_v *
                filter_strength
            )

            # =================================================
            # DISTANCE FROM CENTER
            # =================================================

            difference_h = (
                filtered_h -
                CENTER_H
            )

            difference_v = (
                filtered_v -
                CENTER_V
            )

            # =================================================
            # DEAD ZONE
            # =================================================

            if abs(difference_h) < dead_zone_x:
                difference_h = 0

            if abs(difference_v) < dead_zone_y:
                difference_v = 0

            # =================================================
            # CURSOR POSITION
            # =================================================

            screen_x = (
                0.5 +
                difference_h *
                horizontal_sensitivity
            )

            screen_y = (
                0.5 +
                difference_v *
                vertical_sensitivity
            )

            # Keep inside screen
            screen_x = max(
                0.02,
                min(0.98, screen_x)
            )

            screen_y = max(
                0.02,
                min(0.98, screen_y)
            )

            target_x = int(
                screen_x *
                screen_width
            )

            target_y = int(
                screen_y *
                screen_height
            )

            # =================================================
            # SMOOTH CURSOR
            # =================================================

            current_x += (
                target_x -
                current_x
            ) * smooth_factor

            current_y += (
                target_y -
                current_y
            ) * smooth_factor

            # =================================================
            # MOVE CURSOR
            # =================================================

            pyautogui.moveTo(
                int(current_x),
                int(current_y),
                duration=0
            )

            # =================================================
            # SCROLL
            # =================================================

            now = time.time()

            if (
                screen_y < scroll_zone_top
                and
                now - last_scroll > scroll_delay
            ):

                pyautogui.scroll(
                    scroll_amount
                )

                last_scroll = now

                status = "SCROLLING UP"

            elif (
                screen_y > scroll_zone_bottom
                and
                now - last_scroll > scroll_delay
            ):

                pyautogui.scroll(
                    -scroll_amount
                )

                last_scroll = now

                status = "SCROLLING DOWN"

            else:

                status = "EYES ACTIVE"

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
                left_eye +
                right_eye
            ) / 2

            # =================================================
            # BLINK CLICK
            # =================================================

            if average_eye < blink_threshold:

                status = "BLINK DETECTED"

                # Start timing the blink
                if blink_start is None:
                    blink_start = time.time()

                blink_duration = (
                    time.time() -
                    blink_start
                )

                # =================================================
                # ONE BLINK = ONE CLICK
                # =================================================

                if (
                    blink_duration >= 0.08
                    and
                    not blink_clicked
                    and
                    time.time() -
                    last_click >=
                    click_cooldown
                ):

                    pyautogui.click()

                    print("LEFT CLICK")

                    last_click = time.time()

                    # Lock clicking until eyes open again
                    blink_clicked = True

            else:

                # =================================================
                # EYES OPEN
                # Ready for the next blink
                # =================================================

                blink_start = None
                blink_clicked = False

            # =================================================
            # DRAW IRIS
            # =================================================

            left_x = int(
                left_iris.x *
                width
            )

            left_y = int(
                left_iris.y *
                height
            )

            right_x = int(
                right_iris.x *
                width
            )

            right_y = int(
                right_iris.y *
                height
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

            # =================================================
            # INFORMATION
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
                f"H: {filtered_h:.3f}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"V: {filtered_v:.3f}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "EYE = CURSOR | BLINK = CLICK",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ====================================================
        # CAMERA WINDOW
        # ====================================================

        cv2.imshow(
            "SMART EYE-GAZE",
            frame
        )

        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print()
print("==============================================")
print(" SMART EYE-GAZE CLOSED")
print("==============================================")
