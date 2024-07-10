# import cv2
# import mediapipe as mp
# import time

# # Initialize mediapipe modules and drawing specifications
# mp_drawing = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# # Check if FACE_CONNECTIONS are available
# if hasattr(mp_face_mesh, 'FACEMESH_TESSELATION'):
#     FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION
# else:
#     # Define a basic set of connections if not available
#     FACE_CONNECTIONS = []  # You can define your own connections here if necessary

# # Capture video
# video = cv2.VideoCapture("peoplevideo.mp4")
# # Get the frame rate of the video
# fps = video.get(cv2.CAP_PROP_FPS)
# frame_delay = int(1000 / fps)  # Calculate delay in milliseconds

# # Face mesh processing
# with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, 
#                            refine_landmarks=True) as face_mesh:
#     while True:
#         start_time = time.time()
        
#         ret, image = video.read()
#         if not ret:
#             break
        
#         # Convert color space and set image to non-writable
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
        
#         # Resize image for processing
#         resized_image = cv2.resize(image, (800, 600))
        
#         # Process the image and detect face landmarks
#         results = face_mesh.process(resized_image)
        
#         # Convert image back to writable and BGR color space
#         image.flags.writeable = True
#         image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        
#         # Draw face landmarks and connections if any faces are detected
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=FACE_CONNECTIONS,  # Use predefined or manually defined connections
#                     landmark_drawing_spec=drawing_spec,
#                     connection_drawing_spec=drawing_spec)
        
#         # Display the image
#         cv2.imshow("image", image)
        
#         # Calculate the time taken for processing the frame
#         processing_time = (time.time() - start_time) * 1000
        
#         # Calculate the actual delay time needed to maintain the original video speed
#         actual_delay = max(1, int(frame_delay - processing_time))
        
#         # Wait for the calculated delay time or break if 'q' is pressed
#         if cv2.waitKey(actual_delay) == ord('q'):
#             break
    
#     # Release the video capture and destroy all windows
#     video.release()
#     cv2.destroyAllWindows()
# /////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////
# //////////////////////////BlazeFace (full-range) ////////////////////
# /////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////


import cv2
import mediapipe as mp
import time

# Initialize mediapipe modules and drawing specifications
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection  # Use face_detection module for full-range detection
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Capture video
video = cv2.VideoCapture("longrange.mp4")
# Get the frame rate of the video
fps = video.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Calculate delay in milliseconds

# Face detection and mesh processing
# used full range model with model_selection: 0 or 1. 0. 0= shortrange (2 meter) and 1 = long range (5 meter)
with mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=1) as face_detection:
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) as face_mesh:
        while True:
            start_time = time.time()
            
            ret, image = video.read()
            if not ret:
                break
            
            # Convert color space and set image to non-writable
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Resize image for processing
            resized_image = cv2.resize(image, (800, 600))
            
            # Detect faces in the image
            face_detection_results = face_detection.process(resized_image)
            
            # Convert image back to writable and BGR color space
            image.flags.writeable = True
            image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
            
            # If faces are detected, process each face with face mesh
            if face_detection_results.detections:
                for detection in face_detection_results.detections:
                    # Extract bounding box and landmarks from detection
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Ensure bounding box is within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(iw - x, w)
                    h = min(ih - y, h)
                    
                    # Crop and resize the detected face area for face mesh processing
                    if w > 0 and h > 0:  # Ensure width and height are positive
                        face_image = image[y:y + h, x:x + w]
                        face_image_resized = cv2.resize(face_image, (800, 600))
                        face_image_rgb = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2RGB)
                        
                        # Process the face area to detect face landmarks
                        face_mesh_results = face_mesh.process(face_image_rgb)
                        
                        # Draw landmarks if detected
                        if face_mesh_results.multi_face_landmarks:
                            for face_landmarks in face_mesh_results.multi_face_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=face_image,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=drawing_spec,
                                    connection_drawing_spec=drawing_spec)
            
            # Display the image
            cv2.imshow("image", image)
            
            # Calculate the time taken for processing the frame
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate the actual delay time needed to maintain the original video speed
            actual_delay = max(1, int(frame_delay - processing_time))
            
            # Wait for the calculated delay time or break if 'q' is pressed
            if cv2.waitKey(actual_delay) == ord('q'):
                break
        
        # Release the video capture and destroy all windows
        video.release()
        cv2.destroyAllWindows()
