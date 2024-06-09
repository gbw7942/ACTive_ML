import cv2
import threading
import time

# URL of the video stream
video_url = "https://iusopen.ezvizlife.com/v3/openlive/L37100666_1_1.m3u8?expire=1778165089&id=711710851078774784&c=82f56de598&t=2096574bdff8f10b8b8e944315f6b97f935e8887c6bddbd73ecdcf46aac1e09d&ev=100"

# Open the video stream
cap = cv2.VideoCapture(video_url)

def capture_frame():
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Save frame as JPEG file
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = f'./train/frame_{timestamp}.jpg'
            filename = f"frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        else:
            print("Failed to capture frame")
    threading.Timer(0.5, capture_frame).start()

# Start capturing frames every second
capture_frame()
