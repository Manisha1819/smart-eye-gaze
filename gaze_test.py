import cv2
import mediapipe as mp
import json

# --------------------------------
# Load calibration
# --------------------------------

with open("calibration.json", "r") as file:
    calibration = json.load(file)

CENTER_H = calibration["CENTER"]["h"]
CENTER_V = calibration["CENTER"]["v"]

print("Center calibration:")
print("H =", CENTER_H)
print("V =", CENTER_V)

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

print("Eye direction test started.")
print("Press Q to quit.")

# --------------------------------
# Start face landmarker
# --------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect face
        result = landmarker.detect(mp_image)

        direction = "FACE NOT DETECTED"

        if result.face_landmarks:

            face = result.face_landmarks[0]

            # --------------------------------
            # Iris landmarks
            # --------------------------------

            left_iris = face[468]
            right_iris = face[473]

            # --------------------------------
            # Eye landmarks
            # --------------------------------

            left_outer = face[33]
            left_inner = face[133]

            right_outer = face[362]
            right_inner = face[263]

            left_top = face[159]
            left_bottom = face[145]

            right_top = face[386]
            right_bottom = face[374]

            # --------------------------------
            # Horizontal position
            # --------------------------------

            left_eye_width = abs(
                left_inner.x - left_outer.x
            )

            right_eye_width = abs(
                right_inner.x - right_outer.x
            )

            if (
                left_eye_width > 0.01
                and right_eye_width > 0.01
            ):

                left_h = (
                    left_iris.x
                    - min(left_outer.x, left_inner.x)
                ) / left_eye_width

                right_h = (
                    right_iris.x
                    - min(right_outer.x, right_inner.x)
                ) / right_eye_width

                horizontal = (
                    left_h + right_h
                ) / 2

            else:
                horizontal = CENTER_H

            # --------------------------------
            # Vertical position
            # --------------------------------

            left_eye_height = abs(
                left_bottom.y - left_top.y
            )

            right_eye_height = abs(
                right_bottom.y - right_top.y
            )

            if (
                left_eye_height > 0.01
                and right_eye_height > 0.01
            ):

                left_v = (
                    left_iris.y
                    - min(left_top.y, left_bottom.y)
                ) / left_eye_height

                right_v = (
                    right_iris.y
                    - min(right_top.y, right_bottom.y)
                ) / right_eye_height

                vertical = (
                    left_v + right_v
                ) / 2

            else:
                vertical = CENTER_V

            # --------------------------------
            # Difference from center
            # --------------------------------

            h_difference = (
                horizontal - CENTER_H
            )

            v_difference = (
                vertical - CENTER_V
            )

            # --------------------------------
            # Direction detection
            # --------------------------------

            direction = "LOOKING CENTER"

            if h_difference < -0.04:

                direction = "LOOKING LEFT"

            elif h_difference > 0.04:

                direction = "LOOKING RIGHT"

            elif v_difference < -0.04:

                direction = "LOOKING UP"

            elif v_difference > 0.04:

                direction = "LOOKING DOWN"

            # --------------------------------
            # Draw iris points
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

            # --------------------------------
            # Display values
            # --------------------------------

            cv2.putText(
                frame,
                f"H = {horizontal:.3f}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"V = {vertical:.3f}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"dH = {h_difference:.3f}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"dV = {v_difference:.3f}",
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # --------------------------------
        # Display direction
        # --------------------------------

        cv2.putText(
            frame,
            direction,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.imshow(
            "Smart Eye-Gaze Direction",
            frame
        )

        # --------------------------------
        # Quit
        # --------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# --------------------------------
# Close camera
# --------------------------------

camera.release()
cv2.destroyAllWindows()

print("Camera closed.")