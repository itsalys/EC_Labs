import cv2
import face_recognition # Refer to https://github.com/ageitgey/face_recognition



#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame
        ret, frame = cap.read()  
        
        # resizing for faster detection
        frame = cv2.resize(frame, (256, 256)) #Uncomment and see the speed up
        
        #Extract face locations using the face_recogniton library
        face_locations = face_recognition.face_locations(frame)
              
        #Draw a rectangle around the face
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
            # Extract the landmarks based on the standard face model
            landmarks = face_recognition.face_landmarks(frame,[face_location])[0]
            
            #Draw circles for each facial landmark
            for landmark_type, landmark_points in landmarks.items():
                for point in landmark_points:
                    cv2.circle(frame,point,2,(0,0,255),-1)
        
        
        cv2.imshow("frame",frame)    
        
        if cv2.waitKey(10) == ord('q'):
           break
        
    
    except KeyboardInterrupt:
        break


cap.release()
cv2.destroyAllWindows()
