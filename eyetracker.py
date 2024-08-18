import cv2 
import mediapipe
import pyautogui

# Initialize face mesh landmarks detector
face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Capture from the webcam
cam = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

while True:
    ret, image = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    image = cv2.flip(image, 1)
    window_h, window_w, _ = image.shape

    # Convert the image to RGB for face mesh processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_image)

    # Get the landmarks for all detected faces
    all_face_landmark_points = processed_image.multi_face_landmarks
    
    # Process the first face (if any are detected)
    if all_face_landmark_points:
        one_face_landmark_points = all_face_landmark_points[0].landmark  # Corrected variable name
        for id, landmark_points in enumerate(one_face_landmark_points[474:478]):  # Get eye positions
            x = int(landmark_points.x * window_w)
            y = int(landmark_points.y * window_h)
            if id == 1:  # Move mouse based on eye position
                mouse_x = int(screen_w / window_w * x)
                mouse_y = int(screen_h / window_h * y)
                pyautogui.moveTo(mouse_x, mouse_y) 

            cv2.circle(image, (x, y), 3, (0, 0, 255))

        # Left eye clicker
        left_eye = [one_face_landmark_points[145], one_face_landmark_points[159]]  # Corrected variable name
        for landmark_points in left_eye:
            x = int(landmark_points.x * window_w)
            y = int(landmark_points.y * window_h)
            cv2.circle(image, (x, y), 3, (0, 225, 255))

        # Check for click (eye blink)
        if (left_eye[0].y - left_eye[1].y) < 0.0135:
            pyautogui.click()  
            pyautogui.sleep(1)
            print('Mouse clicked')

    cv2.imshow("Eye controlled mouse", image)
    key = cv2.waitKey(1)  # Continuously capture image
    if key == 27:  # Escape key
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()

