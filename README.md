# Smart Eye-Gaze Based Hands-Free Reading & Navigation System

## 📌 Overview

Smart Eye-Gaze is a computer-vision-based hands-free computer interaction system that allows users to control basic computer operations using eye movements and blinks.

The system is designed to reduce dependence on a traditional mouse and keyboard by providing eye-controlled cursor movement, scrolling, and blink-based clicking.

## ✨ Features

- 👁️ Eye-controlled cursor movement
- 🖱️ Blink-to-click interaction
- 📜 Eye-controlled page scrolling
- 🎯 Personalized eye-gaze calibration
- 👤 Face and iris tracking
- 🎛️ Simple graphical control panel
- ⏹️ Start, stop, calibrate, and exit controls
- ⚡ Real-time computer interaction

## 🧠 How It Works

The system uses a webcam to capture the user's face and eye movements.

MediaPipe Face Landmarker detects facial landmarks and iris positions. The system processes these landmarks to estimate the user's horizontal and vertical gaze position.

The estimated gaze position is converted into screen coordinates and used to control the mouse cursor.

Different gaze regions are used for scrolling:

- Looking toward the top → Scroll up
- Looking toward the bottom → Scroll down

The system also analyzes eye openness to detect a blink. A deliberate blink is converted into a single left-click.

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- Tkinter
- Computer Vision
- Face & Iris Landmark Detection

## 📂 Project Structure

```text
smart-eye-gaze/
│
├── models/
│   └── face_landmarker.task
│
├── smart_eye_app.py
├── smart_eye_gaze.py
├── calibrate.py
├── calibration.json
└── README.md
```

## ⚙️ Requirements

Python 3.x is required.

Install the required packages using:

```bash
pip install opencv-python mediapipe pyautogui
```

## 🚀 Running the Project

### 1. Start the Control Panel

```bash
python smart_eye_app.py
```

### 2. Calibrate

Use the **CALIBRATE** button to create personalized gaze calibration data.

### 3. Start the System

Click **START** from the control panel.

The system will activate the webcam and load the saved calibration.

## 🎮 Controls

| Action | Function |
|---|---|
| Move eyes | Move cursor |
| Look toward top | Scroll up |
| Look toward bottom | Scroll down |
| Blink | Left click |
| Press Q | Exit |

## 🎯 Calibration

Calibration establishes the user's natural center gaze position.

During calibration:

1. Keep your head still.
2. Move only your eyes.
3. Follow the calibration instructions.
4. The resulting center position is saved in `calibration.json`.

## ♿ Accessibility

The project explores hands-free computer interaction for users who may have difficulty using conventional mouse or keyboard input.

By combining gaze-based cursor movement, scrolling, and blink-based clicking, the system provides an alternative interaction method for basic computer navigation.

## 🔮 Future Improvements

Possible future improvements include:

- More accurate gaze estimation
- Adaptive calibration
- Double-blink interaction
- Dwell-based clicking
- Improved head-movement compensation
- Additional accessibility controls
- Multi-screen support
- Better performance under different lighting conditions

## 👩‍💻 Project

**Smart Eye-Gaze Based Hands-Free Reading & Navigation System**

Developed using Python, Computer Vision, MediaPipe, and PyAutoGUI.

