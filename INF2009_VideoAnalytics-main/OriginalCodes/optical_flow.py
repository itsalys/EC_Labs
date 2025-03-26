#%% OpenCV based real-time optical flow estimation and tracking
# Ref: https://github.com/daisukelab/cv_opt_flow/tree/master
import numpy as np
import cv2

#%% Generic Parameters
color = np.random.randint(0,255,(100,3)) # Create some random colors


#%% Parameters for Lucas Kanade optical flow approach [Ref: https://cseweb.ucsd.edu//classes/sp02/cse252/lucaskanade81.pdf]
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                       maxLevel = 2,
                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


#%% Flow estimation is always with respect to previous frame and the below code is required to be done for the first time as called from main
def set1stFrame(frame):
    
    # Converting to gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params) # Corner detection using https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)
    
    return frame_gray,mask,p0

#%% Lucas Kanade optical flow approach [Ref: https://cseweb.ucsd.edu//classes/sp02/cse252/lucaskanade81.pdf]
def LucasKanadeOpticalFlow (frame,old_gray,mask,p0):
    
    # Converting to gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    # calculate optical flow
    if (p0 is None or len(p0) ==0):
        p0 = np.array([[50, 50], [100, 100]], dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,
                                           p0, None, **lk_params)
    
    if p1 is not None:    
    
        # Select good points (skip no points to avoid errors)
        good_new = p1[st==1]
        good_old = p0[st==1]
    
        # draw the tracks
        for i, (new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), color[i].tolist(), 2)
            frame_gray = cv2.circle(frame_gray, (int(a),int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
    
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
    return img,old_gray,p0


#%% Computes a dense optical flow using the Gunnar Farneback's algorithm.
step = 16 

def DenseOpticalFlowByLines(frame, old_gray):
    
    # Converting to gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    
    h, w = frame_gray.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1)
    
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)  # https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af 
    
    fx, fy = flow[y,x].T
    
    # Plot the streamlines
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)    
    cv2.polylines(frame, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)
    return frame


#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

firstframeflag = 1

# The loop will break on pressing the 'q' key
while True:
    try:
        
        if (firstframeflag):
            # Capture one frame
            ret, frame = cap.read() 
            
            old_gray,mask,p0 = set1stFrame(frame)            
          
            firstframeflag = 0
        
        # Capture one frame
        ret, frame = cap.read()  
        
        img = DenseOpticalFlowByLines(frame, old_gray)
        
        #img,old_gray,p0 = LucasKanadeOpticalFlow(frame,old_gray,mask,p0)
        
        cv2.imshow("Optical Flow", img)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
