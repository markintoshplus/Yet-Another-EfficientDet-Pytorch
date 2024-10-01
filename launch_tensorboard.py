import os
import subprocess
import time
import webbrowser
from datetime import datetime

# Path to your logs
log_dir = "logs/taco/tensorboard"

# Get all subdirectories
subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

# Sort subdirectories by creation time (most recent first)
latest_dir = max(subdirs, key=os.path.getctime)

# Start TensorBoard process
tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", latest_dir])

# Wait a moment for TensorBoard to start
time.sleep(3)

# Open the default web browser
webbrowser.open("http://localhost:6006")

print(f"TensorBoard started with log directory: {latest_dir}")
print("TensorBoard interface opened in your default web browser.")
print("Press Ctrl+C to stop TensorBoard when you're done.")

try:
    # Keep the script running
    tensorboard_process.wait()
except KeyboardInterrupt:
    # Handle Ctrl+C to stop TensorBoard gracefully
    print("\nStopping TensorBoard...")
    tensorboard_process.terminate()
    tensorboard_process.wait()
    print("TensorBoard stopped.")