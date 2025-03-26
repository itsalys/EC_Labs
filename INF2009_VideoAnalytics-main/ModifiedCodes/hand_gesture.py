#%% Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/raspberry_pi/
# Download hand gesture detector model wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python # import the python wrapper
from mediapipe.tasks.python import vision # import the API for calling the recognizer and setting parameters
from mediapipe.framework.formats import landmark_pb2 #The base land mark atlas
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



#%% Parameters
numHands = 2 # Number of hands to be detected
model = 'gesture_recognizer.task' # Model for hand gesture detection Download using wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
minHandDetectionConfidence = 0.5 # Thresholds for detecting the hand
minHandPresenceConfidence = 0.5
minTrackingConfidence = 0.5
frameWidth = 640
frameHeight = 480

# Visualization parameters
row_size = 50  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 0)  # black
font_size = 1
font_thickness = 1

# Label box parameters
label_text_color = (255, 255, 255)  # white
label_font_size = 1
label_thickness = 2

#%% Initializing results and save result call back for appending results.
recognition_frame = None
recognition_result_list = []

def save_result(result: vision.GestureRecognizerResult,
                unused_output_image: mp.Image,timestamp_ms: int):
    
    recognition_result_list.append(result)

#%% Create an Hand Gesture Control object.
# Initialize the gesture recognizer model
base_options = python.BaseOptions(model_asset_path=model)
options = vision.GestureRecognizerOptions(base_options=base_options,
                                        running_mode=vision.RunningMode.LIVE_STREAM,
                                        num_hands=numHands,
                                        min_hand_detection_confidence=minHandDetectionConfidence,
                                        min_hand_presence_confidence=minHandPresenceConfidence,
                                        min_tracking_confidence=minTrackingConfidence,
                                        result_callback=save_result)
recognizer = vision.GestureRecognizer.create_from_options(options)

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
        
        # Run hand landmarker using the model.        
        recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)
        
        if recognition_result_list:
          
          # Draw landmarks and write the text for each hand.
          for hand_index, hand_landmarks in enumerate(
              recognition_result_list[0].hand_landmarks):
           
            # Calculate the bounding box of the hand
            x_min = min([landmark.x for landmark in hand_landmarks])
            y_min = min([landmark.y for landmark in hand_landmarks])
            y_max = max([landmark.y for landmark in hand_landmarks])
    
            # Convert normalized coordinates to pixel values
            frame_height, frame_width = current_frame.shape[:2]
            x_min_px = int(x_min * frame_width)
            y_min_px = int(y_min * frame_height)
            y_max_px = int(y_max * frame_height)
    
            # Get gesture classification results
            if recognition_result_list[0].gestures:
              gesture = recognition_result_list[0].gestures[hand_index]
              category_name = gesture[0].category_name
              score = round(gesture[0].score, 2)
              result_text = f'{category_name} ({score})'
    
              # Compute text size
              text_size = \
              cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                              label_thickness)[0]
              text_width, text_height = text_size
    
              # Calculate text position (above the hand)
              text_x = x_min_px
              text_y = y_min_px - 10  # Adjust this value as needed
    
              # Make sure the text is within the frame boundaries
              if text_y < 0:
                text_y = y_max_px + text_height
    
              # Draw the text
              cv2.putText(current_frame, result_text, (text_x, text_y),
                          cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                          label_text_color, label_thickness, cv2.LINE_AA)
    
            # Draw hand landmarks on the frame using the atlas
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                              z=landmark.z) for landmark in
              hand_landmarks
            ])
            mp_drawing.draw_landmarks(
              current_frame,
              hand_landmarks_proto,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
    
          recognition_frame = current_frame
          recognition_result_list.clear()
    
        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()