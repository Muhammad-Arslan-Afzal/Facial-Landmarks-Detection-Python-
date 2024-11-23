
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

frontal_face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
profile_face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

video_path = "freepeoplewalking.mp4"
cap = cv2.VideoCapture(video_path)
skip_frames = 10  
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % skip_frames != 0:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frontal_faces = frontal_face_classifier.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=9, minSize=(50, 50)
    )
    profile_faces = profile_face_classifier.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=9, minSize=(50, 50)
    )

    for (x, y, w, h) in frontal_faces:
        face_roi = frame[y:y + h, x:x + w]
        rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_face_roi)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(face_roi, landmarks)
        frame[y:y + h, x:x + w] = face_roi
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in profile_faces:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        face_roi = frame[y:y + h, x:x + w]
        rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_face_roi)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(face_roi, landmarks)
        frame[y:y + h, x:x + w] = face_roi

    frame_resized = cv2.resize(frame, (960, 540))
    cv2.imshow("Video with Face Detection and Landmarks", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

