**Video Analytics with Raspberry Pi using Web Camera**

**Objective:** By the end of this session, participants will understand how to set up a web camera with the Raspberry Pi, capture video streams, and perform basic and advanced video analytics.

---

**Prerequisites:**
1. Raspberry Pi with Raspbian OS installed.
2. MicroSD card (16GB or more recommended).
3. Web camera compatible with Raspberry Pi.
4. Internet connectivity (Wi-Fi or Ethernet).
5. Basic knowledge of Python and Linux commands.

---

**1. Introduction (10 minutes)**
- Video analytics is an emerging field employed to extract valuable insights from video data. Edge video analytics with real-time processing capabilities is chellenging but important and inevitable due to privacy/security concerns. Also, in many cases redundancy can be avoided to save on the bandwidth requirements (e.g. compress the video to have only key (important) frames). In this lab, few basic and advanced video processing tasks on edge devices is introduced. An overview of the experiments/setup is as follows:
![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/882c84dc-1989-4039-807d-554a079e3776)

**2. Setting up the Raspberry Pi (10 minutes)**
- Booting up the Raspberry Pi.
- Setting up Wi-Fi/Ethernet.
- System updates:
  ```bash
  sudo apt update
  sudo apt upgrade
  ```

**3. Connecting and Testing the Web Camera (5 minutes)**
- Please ensure the web camera is working and proceed to subsequent steps.

**4. Introduction to real-time video processing on raspberry pi (20 minutes)**
- **[Important!] Set up and activate a virtual environment named "video" for this experiment (to avoid conflicts in libraries) as below. You can also reuse the virtual environment "image" as we are employing opencv and mediapipe libraries for video analytics**
  ```bash
  sudo apt install python3-venv
  python3 -m venv video
  source video/bin/activate
- Installing OpenCV:
  ```bash
  pip install opencv-python  
  ```
- [Optical flow](https://en.wikipedia.org/wiki/Optical_flow) estimation is employed to track moving objects in a video sequence. In this section, we will employ the purely opencv based [sample code](Codes/optical_flow.py) for estimaging the flow using Lucas Kanade Optical Flow approach and Flow Farneback approach. The displays are in the form of streamlines or directional arrows as shown below. \
  ![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/c5987191-27ff-44f9-ac85-d1a673477dc8) 
  ![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/f9a6d18e-4973-4af9-80f5-45901d090cc1)
  - **[Important]** You need to comment/uncomment respective lines (line 119/121) to activate the desired results. Modify the parameters (line 12/18) by looking into the OpenCV documentation and observe/note down the observations/conclusions.

**5. Advanced Video Analytics (40 minutes)**
- We will employ a light weight opensource library named *"Mediapipe"* for tasks such as face landmark detection, pose estimation, hand landmark detection, hand gesture recognition and object detection using pretrained neural network models.
- [MediaPipe](https://developers.google.com/mediapipe) is a on-device (*embedded machine learning*) framework for building cross platform multimodal applied ML pipelines that consist of fast ML inference, classic computer vision, and media processing (e.g. video decoding). MediaPipe was open sourced at CVPR in June 2019 as v0.5.0 and has various lightweight models developed with Tensorflow lite available for usage.
- Installing media pipe:
  ```bash  
  pip install mediapipe
  ```
- Hand landmark detection
  - Download the handlandmark detection model:
    ```bash
    wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    ``` 
  - The [sample code](Codes/hand_landmark.py) employs opencv and mediapipe to detect the human hand and subsequently the finger locations (the tip of thumb and index finger as well as a simple logic to predict if the thumb is pointing up) based on the [finger model](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) outlined below :
    ![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/1090e213-7a56-4059-9386-50123bd6f8f8)
  - Modify the code to show all the 21 finger points and observe the same while moving the hand.
  - Modify the code to predict the number of fingers and display the same overlaid on the image as text (e.g. if four fingers are raised, display '4' on the screen and if three fingers on one hand and two on the other, the display should be '5').

**6. Advanced Video Analytics (20 minutes)**
- In this section, we will work on more advanced analytics tasks such as hand gesture recognition and object detection based on pretrianed light weight models.
- Hand gesture recognition
  - Download the hand gesture recognition model:
    ```bash
     wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
    ``` 
  - The [sample code](Codes/hand_gesture.py) shows a real-time hand gesture recongition task. A sample snapshot of the code result for victory sign is shown below: \
    ![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/84bf1517-22c0-427a-9ca7-047551f1b50e)
- Object detection
  - Download the light weight EfficientDet object detection model:
    ```bash
     wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
    ``` 
  - The [sample code](Codes/obj_detection.py) shows a real-time object detection task.
  - Based on the above code, write a code to do object detection based video summarization (e.g. for a video with only frames having a cellphone) 
    
---

**[Optional] Homework/Extended Activities:**
1. Experiment with more advanced tracking algorithms available in OpenCV.
2. Build a gesture based video player control (e.g. could use libraries like [Pyautogui](https://pyautogui.readthedocs.io/en/latest/) for the same) 
3. Build a surveillance system based on video based motion detection.

---

**Resources:**
1. Raspberry Pi official documentation.
2. OpenCV documentation and tutorials.
3. Mediapipe documentation and tutorials.

---
