#%% Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/raspberry_pi
import cv2
import mediapipe as mp
import time
import os

from mediapipe.tasks import python  # import the python wrapper
from mediapipe.tasks.python import vision  # import the API for calling the recognizer and setting parameters

#%% Parameters
maxResults = 5
scoreThreshold = 0.25
frameWidth = 640
frameHeight = 480
model = 'efficientdet.tflite'
object_of_interest = "cell phone"  # Change this to the object you want to filter

# Video saving parameters
output_filename = "summarized_video.mp4"
fps = 20  # Frames per second for the output video
frame_count = 0  # Track number of frames written

# Visualization parameters
MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # Black

#%% Initialize results list
detection_result_list = []

def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    detection_result_list.append(result)

#%% Create object detector
base_options = python.BaseOptions(model_asset_path=model)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.LIVE_STREAM,
                                       max_results=maxResults, 
                                       score_threshold=scoreThreshold,
                                       result_callback=save_result)
detector = vision.ObjectDetector.create_from_options(options)

#%% OpenCV Video Capture and Output Video Writer
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Initialize video writer for summarization
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frameWidth, frameHeight))

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip for correct orientation
        
        # Convert the image to RGB for model inference
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        detection_found = False  # Flag to check if relevant object is detected

        if detection_result_list:
            for detection in detection_result_list[0].detections:
                # Get bounding box coordinates
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

                # Extract object name and confidence score
                category = detection.categories[0]
                category_name = category.category_name.lower()
                probability = round(category.score, 2)
                result_text = f"{category_name} ({probability})"
                text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)

                # Draw bounding box and label
                cv2.rectangle(frame, start_point, end_point, (0, 165, 255), 3)  # Orange box
                cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

                # If the object of interest is detected, mark it for saving
                if object_of_interest in category_name:
                    print("detected !")
                    out.write(frame)
                    frame_count += 1
                    detection_found = True

            detection_result_list.clear()

        # Display the annotated frame
        cv2.imshow('Object Detection - Video Summarization', frame)

        # # Ensure that at least one frame is written
        # if detection_found:
        #     out.write(frame)
        #     frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break

# Release resources
cap.release()
if frame_count > 0:
    out.release()
    print(f"Summarized video saved as {output_filename} with {frame_count} frames.")
else:
    out.release()
    os.remove(output_filename)  # Delete empty file if no frames were saved
    print("No relevant frames detected. Video file not saved.")

cv2.destroyAllWindows()
