# Reference: https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/
import cv2
import numpy as np

#%% Defining a list of boundaries in the RGB color space 
# (or rather, BGR, since OpenCV represents images as NumPy arrays in reverse order) 
# Refer to https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
boundaries = [
	([0, 0, 90], [90, 90, 255]), # For Red
	([90, 0, 0], [255, 90, 90]), # For Blue
	([0, 90, 0], [90, 255, 90]), # For Green
        ([0, 180, 180], [100, 255, 255]) ] # For Yellow 

#%% Normalize the Image for display (Optional)
def normalizeImg (Img):
    Img= np.float64(Img) #Converting to float to avoid errors due to division
    norm_img = (Img - np.min(Img))/(np.max(Img) - np.min(Img))
    norm_img = np.uint8(norm_img*255.0)
    return norm_img
    
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
        
        output=[]
        
        # loop over the boundaries       
        for (lower, upper) in boundaries:
        	# create NumPy arrays from the boundaries
        	lower = np.array(lower, dtype = "uint8")
        	upper = np.array(upper, dtype = "uint8")
        	
            # find the colors within the specified boundaries and apply the mask (basically segmenting for colours)
        	mask = cv2.inRange(frame, lower, upper)
        	output.append(cv2.bitwise_and(frame, frame, mask = mask)) #Segmented frames are appended
        
        # Output is appeneded to be of size Pixels X 3 (for R, G, B)
        red_img = normalizeImg(output[0])
        green_img = normalizeImg(output[1])
        blue_img = normalizeImg(output[2])
        yellow_img = normalizeImg(output[3])  # Yellow mask processing
       
        # horizontal Concatination for displaying the images and colour segmentations
        catImg = cv2.hconcat([frame,red_img,green_img,blue_img,yellow_img])
        cv2.imshow("Images with Colours",catImg)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
