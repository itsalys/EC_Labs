import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#%% Parameters
numHands = 2  # Number of hands to be detected
model = 'hand_landmarker.task'  # Model for finding the hand landmarks
minHandDetectionConfidence = 0.5  # Thresholds for detecting the hand
minHandPresenceConfidence = 0.5
minTrackingConfidence = 0.5
frameWidth = 640
frameHeight = 480

# Visualization parameters
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

#%% Create a HandLandmarker object.
base_options = python.BaseOptions(model_asset_path=model)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=numHands,
    min_hand_detection_confidence=minHandDetectionConfidence,
    min_hand_presence_confidence=minHandPresenceConfidence,
    min_tracking_confidence=minTrackingConfidence)
detector = vision.HandLandmarker.create_from_options(options)

#%% OpenCV Video Capture and frame analysis
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
        frame = cv2.flip(frame, 1)  # Flip image to match camera orientation

        # Convert the image from BGR to RGB as required by the model
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run hand landmarker
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = detector.detect(mp_image)

        hand_landmarks_list = detection_result.hand_landmarks
        total_fingers = 0  # Store total fingers from both hands

        # Loop through detected hands
        for hand_landmarks in hand_landmarks_list:
            # Count fingers for the current hand
            fingers = 0
            if hand_landmarks[4].x > hand_landmarks[2].x:
                fingers += 1  # Thumb extended
            
            finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
            finger_pips = [6, 10, 14, 18]  # Corresponding PIP joints
            
            for tip, pip in zip(finger_tips, finger_pips):
                if hand_landmarks[tip].y < hand_landmarks[pip].y:
                    fingers += 1  # Finger is extended
            
            total_fingers += fingers  # Sum the total fingers detected

            # Draw circles at each of the 21 hand landmark points
            for i in range(21):  
                x = int(hand_landmarks[i].x * frame.shape[1])
                y = int(hand_landmarks[i].y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Display total number of fingers detected
        cv2.putText(frame, f'Total Fingers: {total_fingers}', (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
        cv2.imshow('Annotated Image', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
