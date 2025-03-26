import cv2
import numpy as np
from skimage import feature # Refer to https://scikit-image.org/ for full capabalities of scikit-image library
from skimage import exposure


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
        #frame = cv2.resize(frame, (256, 256)) #Uncomment and see the speed up
        
        # Converting to gray scal as HOG feature extraction in scikit-image works only on gray scale image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extact the HoG featues from the image
        (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
    	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
    	visualize=True)
        
        # Rescale intensity to be within 0-255 (contrast stretching)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))        
        hogImage = hogImage.astype("uint8")
        
        # Converting gray to RGB
        hogImg = cv2.cvtColor(hogImage,cv2.COLOR_GRAY2RGB)
        
        # Horizontal concatenation to show both input image and its HoG features.
        catImg = cv2.hconcat([frame,hogImg])        
        cv2.imshow("HOG Image", catImg)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
