import cv2 
import mediapipe
import pyautogui

#face outline
face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)

#capture camera
cam = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

while True:
    ret, image = cam.read()
    if not ret:
        print("Failed to capture image")
        break
    image = cv2.flip(image,1)
    window_h,window_w,_ = image.shape
    #use to capture face outline
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_image)
    #capture multiple faces
    all_face_landmark_points = processed_image.multi_face_landmarks
    
    #get first face if ultiple
    if all_face_landmark_points:
        one_face_landark_points = all_face_landmark_points[0].landmark
        for id, landmark_points in enumerate(one_face_landark_points[474:478]): #get pos
            x = int(landmark_points.x * window_w)
            y = int(landmark_points.y * window_h)
            #print(x, y)
            if id ==1: #move mouse with eye
                mouse_x = int(screen_w / window_w * x)
                mouse_y = int(screen_h /  window_h * y)
                pyautogui.moveTo(mouse_x, mouse_y) 

            cv2.circle(image,(x,y),3,(0,0,255))
    #left eye clicker
    left_eye = [one_face_landark_points[145],one_face_landark_points[159]]
    for landmark_points in left_eye:
            x = int(landmark_points.x * window_w)
            y = int(landmark_points.y * window_h)
            #print(x, y)
            cv2.circle(image,(x,y),3,(0,225,255))  
    if(left_eye[0].y - left_eye[1].y <0.0135):
        pyautogui.click()  
        pyautogui.sleep(1)
        print('mouse clicked')
    cv2.imshow("Eye controlled mouse, image", image)
    key = cv2.waitKey(1) #continously capture image
    if key == 27:  # Escape key
        break
#release camera
cam.release()
cv2.destroyAllWindows()

