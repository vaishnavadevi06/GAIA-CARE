import cv2
import pyautogui

from gaze_tracker import GazeTracker
from screen_mapper import ScreenMapper
from smoothing import Smoother

pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)

gaze_tracker = GazeTracker()
screen_mapper = ScreenMapper()
smoother = Smoother(alpha=0.85)

print("Eye Tracking MVP Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gaze, frame = gaze_tracker.get_gaze(frame)

    if gaze:
        smooth_x, smooth_y = smoother.smooth(gaze[0], gaze[1])
        screen_x, screen_y = screen_mapper.map_to_screen(smooth_x, smooth_y)

        # Move cursor (or comment this & draw dot only)
        pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow("Eye Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
