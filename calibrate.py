import cv2
import mediapipe as mp
import json
import time

# -----------------------------
# MediaPipe setup
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# -----------------------------
# Camera
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

# -----------------------------
# Calibration positions
# -----------------------------

positions = [
    "CENTER",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN"
]

calibration = {}

print()
print("====================================")
print("SMART EYE-GAZE CALIBRATION")
print("====================================")
print()
print("Keep your head still.")
print("Move ONLY your eyes.")
print()

with FaceLandmarker.create_from_options(options) as landmarker:

    for position in positions:

        print()
        print("LOOK:", position)
        print("Press SPACE when ready.")

        while True:

            success, frame = camera.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"LOOK {position}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                "Press SPACE",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Smart Eye-Gaze Calibration",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):

                samples = []

                print("Collecting samples...")

                for i in range(40):

                    success, frame = camera.read()

                    if not success:
                        continue

                    frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    result = landmarker.detect(image)

                    if result.face_landmarks:

                        face = result.face_landmarks[0]

                        # -------------------------
                        # Left eye
                        # -------------------------

                        left_iris = face[468]

                        left_outer = face[33]
                        left_inner = face[133]

                        left_top = face[159]
                        left_bottom = face[145]

                        # -------------------------
                        # Right eye
                        # -------------------------

                        right_iris = face[473]

                        right_outer = face[362]
                        right_inner = face[263]

                        right_top = face[386]
                        right_bottom = face[374]

                        # -------------------------
                        # Horizontal normalization
                        # -------------------------

                        left_h = (
                            left_iris.x - left_inner.x
                        ) / (
                            left_outer.x - left_inner.x
                        )

                        right_h = (
                            right_iris.x - right_outer.x
                        ) / (
                            right_inner.x - right_outer.x
                        )

                        horizontal = (
                            left_h + right_h
                        ) / 2

                        # -------------------------
                        # Vertical normalization
                        # -------------------------

                        left_v = (
                            left_iris.y - left_top.y
                        ) / (
                            left_bottom.y - left_top.y
                        )

                        right_v = (
                            right_iris.y - right_top.y
                        ) / (
                            right_bottom.y - right_top.y
                        )

                        vertical = (
                            left_v + right_v
                        ) / 2

                        samples.append(
                            [horizontal, vertical]
                        )

                    time.sleep(0.02)

                if samples:

                    avg_h = (
                        sum(x[0] for x in samples)
                        / len(samples)
                    )

                    avg_v = (
                        sum(x[1] for x in samples)
                        / len(samples)
                    )

                    calibration[position] = {
                        "h": avg_h,
                        "v": avg_v
                    }

                    print(
                        f"{position}: "
                        f"H={avg_h:.3f}, "
                        f"V={avg_v:.3f}"
                    )

                break

            if key == ord("q"):

                camera.release()
                cv2.destroyAllWindows()
                exit()

camera.release()
cv2.destroyAllWindows()

# -----------------------------
# Save calibration
# -----------------------------

with open(
    "calibration.json",
    "w"
) as file:

    json.dump(
        calibration,
        file,
        indent=4
    )

print()
print("====================================")
print("CALIBRATION COMPLETE")
print("Saved: calibration.json")
print("====================================")
print()
print(calibration)