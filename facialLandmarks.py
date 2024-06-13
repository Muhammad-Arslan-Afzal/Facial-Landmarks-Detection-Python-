import cv2
import mediapipe as mp
# Initialize mediapipe modules and drawing specifications
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Check if FACE_CONNECTIONS are available
if hasattr(mp_face_mesh, 'FACEMESH_TESSELATION'):
    FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION
else:
    # Define a basic set of connections if not available
    FACE_CONNECTIONS = []  # You can define your own connections here if necessary

# Capture video
video = cv2.VideoCapture("peoplevideo.mp4")
# Get the frame rate of the video
fps = video.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Calculate delay in milliseconds

# Face mesh processing
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                           refine_landmarks=True) as face_mesh:
    while True:
        ret, image = video.read()
        if not ret:
            break
        
        # Convert color space and set image to non-writable
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Resize image for processing
        resized_image = cv2.resize(image, (800, 600))
        
        # Process the image and detect face landmarks
        results = face_mesh.process(resized_image)
        
        # Convert image back to writable and BGR color space
        image.flags.writeable = True
        image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        
        # Draw face landmarks and connections if any faces are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=FACE_CONNECTIONS,  # Use predefined or manually defined connections
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        
        # Display the image
        cv2.imshow("image", image)
        
        # Wait for the calculated delay time or break if 'q' is pressed
        if cv2.waitKey(frame_delay) == ord('q'):
            break
    
    # Release the video capture and destroy all windows
    video.release()
    cv2.destroyAllWindows()
