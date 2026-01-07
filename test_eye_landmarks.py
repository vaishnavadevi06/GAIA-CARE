import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

IRIS_LANDMARKS = [468, 469, 470, 471]  # Right eye iris

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            # Draw iris landmarks
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_IRISES
            )

            # Get iris center
            h, w, _ = frame.shape
            iris_x = 0
            iris_y = 0

            for idx in IRIS_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                iris_x += int(lm.x * w)
                iris_y += int(lm.y * h)

            iris_x //= len(IRIS_LANDMARKS)
            iris_y //= len(IRIS_LANDMARKS)

            # Draw iris center
            cv2.circle(frame, (iris_x, iris_y), 5, (0, 0, 255), -1)

            print("IRIS DETECTED at:", iris_x, iris_y)

    cv2.imshow("Eye Tracking Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



