import tkinter as tk
import subprocess
import sys

# ============================================================
# SMART EYE-GAZE CONTROL PANEL
# ============================================================

app = tk.Tk()

app.title("Smart Eye-Gaze")
app.geometry("500x550")
app.resizable(False, False)

process = None
calibration_process = None


# ============================================================
# TITLE
# ============================================================

title = tk.Label(
    app,
    text="SMART EYE-GAZE",
    font=("Arial", 26, "bold")
)

title.pack(pady=25)


subtitle = tk.Label(
    app,
    text="Hands-Free Computer Control System",
    font=("Arial", 12)
)

subtitle.pack()


# ============================================================
# STATUS
# ============================================================

status = tk.Label(
    app,
    text="System Ready",
    font=("Arial", 14)
)

status.pack(pady=20)


# ============================================================
# FEATURES
# ============================================================

features = tk.Label(
    app,
    text=
    "✓ Eye-Controlled Cursor\n"
    "✓ Blink-to-Click\n"
    "✓ Eye-Controlled Scrolling\n"
    "✓ Face & Eye Tracking",
    font=("Arial", 13),
    justify="left"
)

features.pack(pady=10)


# ============================================================
# START SYSTEM
# ============================================================

def start_system():

    global process

    if process is None or process.poll() is not None:

        status.config(
            text="Smart Eye-Gaze is RUNNING"
        )

        process = subprocess.Popen(
            [
                sys.executable,
                "smart_eye_gaze.py"
            ]
        )


# ============================================================
# CALIBRATE
# ============================================================

def calibrate_system():

    global calibration_process

    if (
        calibration_process is None
        or
        calibration_process.poll() is not None
    ):

        status.config(
            text="Calibration Started"
        )

        calibration_process = subprocess.Popen(
            [
                sys.executable,
                "calibrate.py"
            ]
        )


# ============================================================
# STOP SYSTEM
# ============================================================

def stop_system():

    global process

    if process is not None:

        process.terminate()
        process = None

        status.config(
            text="Smart Eye-Gaze is STOPPED"
        )


# ============================================================
# START BUTTON
# ============================================================

start_button = tk.Button(
    app,
    text="START",
    command=start_system,
    font=("Arial", 14, "bold"),
    width=18,
    height=2
)

start_button.pack(pady=7)


# ============================================================
# CALIBRATE BUTTON
# ============================================================

calibrate_button = tk.Button(
    app,
    text="CALIBRATE",
    command=calibrate_system,
    font=("Arial", 14, "bold"),
    width=18,
    height=2
)

calibrate_button.pack(pady=7)


# ============================================================
# STOP BUTTON
# ============================================================

stop_button = tk.Button(
    app,
    text="STOP",
    command=stop_system,
    font=("Arial", 14, "bold"),
    width=18,
    height=2
)

stop_button.pack(pady=7)


# ============================================================
# EXIT
# ============================================================

def close_app():

    stop_system()

    if calibration_process is not None:
        try:
            calibration_process.terminate()
        except:
            pass

    app.destroy()


exit_button = tk.Button(
    app,
    text="EXIT",
    command=close_app,
    font=("Arial", 12),
    width=12
)

exit_button.pack(pady=20)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    app,
    text="Computer Vision • AI • Accessibility",
    font=("Arial", 10)
)

footer.pack(
    side="bottom",
    pady=15
)


# ============================================================
# WINDOW CLOSE
# ============================================================

app.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


# ============================================================
# RUN
# ============================================================

app.mainloop()