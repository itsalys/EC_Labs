# For the code to work the Open source Haar Cascade model has to be downloaded and kept in the same folder. 
# Please download the .xml from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

import cv2

# Initiate the Face Detection Cascade Classifier
haarcascade = "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(haarcascade)

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
        frame = cv2.resize(frame, (256, 256)) #Comment and see the speed up
        
        # Converting to gray scale as feature extraction works only on gray scale image
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        # Detect faces using the haarcascade classifier on the "grayscale image"
        faces = detector.detectMultiScale(image_gray)
       
        # returns the bounding boxes for the detected objects and display them using rectangles 
        for face in faces:
            (x,y,w,d) = face
            # Draw a white coloured rectangle around each face using the face's coordinates
            # on the "image_template" with the thickness of 2 
            cv2.rectangle(frame,(x,y),(x+w, y+d),(255, 255, 255), 2)
       
        #resizes the video so its easier to see on the screen
        frame = cv2.resize(frame,(720,720))
        # Display the resulting frame
        cv2.imshow("frame",frame)       
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()