
import cv2
import numpy as np
import mediapipe as mp

# Load the Haar cascade files for face detection
frontal_face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

video_capture = cv2.VideoCapture("freepeoplewalking.mp4")

# Set the desired window size
window_width = 800
window_height = 600

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def overlap(box1, box2):
    # Check if two bounding boxes overlap
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def filter_detections(detections, min_size=40, max_size=400):
    # Filter out detections that are too small or too large
    filtered = []
    for (x, y, w, h) in detections:
        if min_size <= w <= max_size and min_size <= h <= max_size:
            filtered.append((x, y, w, h))
    return filtered

def enhance_face(face_image):
    # Convert to grayscale and apply histogram equalization for illumination enhancement
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    # Convert back to BGR
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    # Sharpening using a kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    
    # Detect frontal faces / camera facing faces
    frontal_faces = frontal_face_classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Filter detections based on size
    frontal_faces = filter_detections(frontal_faces)
    
    # Detect profile faces / side looking faces
    profile_faces = profile_face_classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Filter detections based on size
    profile_faces = filter_detections(profile_faces)
    
    # Filter out profile faces that overlap with frontal faces
    filtered_profile_faces = []
    for pf in profile_faces:
        overlap_found = False
        for ff in frontal_faces:
            if overlap(ff, pf):
                overlap_found = True
                break
        if not overlap_found:
            filtered_profile_faces.append(pf)
    
    return frontal_faces, filtered_profile_faces

def detect_landmarks(face_image):
    # Convert to RGB for MediaPipe
    rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_face_image)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                face_image, landmarks,
                # mp_face_mesh.FACE_CONNECTIONS  # Use FACE_CONNECTIONS if available
            )
    return face_image

while True:
    result, video_frame = video_capture.read()
    if not result:
        break

    frontal_faces, filtered_profile_faces = detect_bounding_box(video_frame)

    # Process each detected face
    for (x, y, w, h) in frontal_faces + filtered_profile_faces:
        # Crop the face
        face_image = video_frame[y:y+h, x:x+w]
        # Enhance the face image
        enhanced_face_image = enhance_face(face_image)
        # Detect and draw landmarks
        face_with_landmarks = detect_landmarks(enhanced_face_image)
        # Replace the cropped face in the original frame with the enhanced face
        video_frame[y:y+h, x:x+w] = face_with_landmarks

    # Resize the video frame to fit the window size
    video_frame_resized = cv2.resize(video_frame, (window_width, window_height))

    cv2.imshow("My Face Detection Project", video_frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()







