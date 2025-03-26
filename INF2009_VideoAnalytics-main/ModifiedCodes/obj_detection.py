#%% Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/raspberry_pi
# Download lightweight ftlite EfficientDet model using wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python # import the python wrapper
from mediapipe.tasks.python import vision # import the API for calling the recognizer and setting parameters


#%% Parameters
maxResults = 5
scoreThreshold = 0.25
frameWidth = 640
frameHeight = 480
model = 'efficientdet.tflite'

# Visualization parameters
MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # black


#%% Initializing results and save result call back for appending results.
detection_frame = None
detection_result_list = []
  
def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
      
      detection_result_list.append(result)

#%% Create an object detection model object.
# Initialize the object detection model
base_options = python.BaseOptions(model_asset_path=model)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.LIVE_STREAM,
                                       max_results=maxResults, score_threshold=scoreThreshold,
                                       result_callback=save_result)
detector = vision.ObjectDetector.create_from_options(options)

#%% Open CV Video Capture and frame analysis (setting the size of the capture resolution as per the model requirements)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame
        ret, frame = cap.read() 
        
        frame = cv2.flip(frame, 1) # To flip the image to match with camera flip
        
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        current_frame = frame
        
        # Run object detection using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        
        if detection_result_list:
            for detection in detection_result_list[0].detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                # Use the orange color for high visibility.
                cv2.rectangle(current_frame, start_point, end_point, (0, 165, 255), 3)
            
                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (MARGIN + bbox.origin_x,
                                 MARGIN + ROW_SIZE + bbox.origin_y)
                cv2.putText(current_frame, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            detection_frame = current_frame
            detection_result_list.clear()
    
        if detection_frame is not None:
            cv2.imshow('object_detection', detection_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()