import pyautogui

class ScreenMapper:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

    def map_to_screen(self, gaze_x, gaze_y):
        screen_x = int(gaze_x * self.screen_w)
        screen_y = int(gaze_y * self.screen_h)
        return screen_x, screen_y
