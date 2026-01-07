import cv2
import mediapipe as mp
import numpy as np


class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,   # REQUIRED for iris
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Correct MediaPipe Iris landmark indices
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

    def get_gaze(self, frame):
        """
        Returns:
            gaze (x, y) normalized between 0–1
            frame with debug drawings
        """

        if frame is None:
            return None, frame

        h, w, _ = frame.shape

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, frame

        face_landmarks = results.multi_face_landmarks[0]

        # Extract iris centers
        left_iris = self._get_iris_center(face_landmarks, self.LEFT_IRIS, w, h)
        right_iris = self._get_iris_center(face_landmarks, self.RIGHT_IRIS, w, h)

        if left_iris is None or right_iris is None:
            return None, frame

        # Average gaze point
        gaze_x = (left_iris[0] + right_iris[0]) / 2
        gaze_y = (left_iris[1] + right_iris[1]) / 2

        # Normalize (0–1)
        gaze_x_norm = gaze_x / w
        gaze_y_norm = gaze_y / h

        # Debug visuals
        cv2.circle(frame, left_iris, 4, (0, 255, 0), -1)
        cv2.circle(frame, right_iris, 4, (0, 255, 0), -1)
        cv2.circle(frame, (int(gaze_x), int(gaze_y)), 6, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"Gaze: {gaze_x_norm:.2f}, {gaze_y_norm:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        return (gaze_x_norm, gaze_y_norm), frame

    def _get_iris_center(self, landmarks, indices, w, h):
        xs = []
        ys = []

        for idx in indices:
            lm = landmarks.landmark[idx]
            xs.append(lm.x * w)
            ys.append(lm.y * h)

        if not xs or not ys:
            return None

        return int(np.mean(xs)), int(np.mean(ys))

