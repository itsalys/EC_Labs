**Image Analytics with Raspberry Pi using Web Camera**

**Objective:** By the end of this session, participants will understand how to set up a web camera with the Raspberry Pi, capture images, and perform basic and advanced image analytics.

---

**Prerequisites:**
1. Raspberry Pi with Raspbian OS installed.
2. MicroSD card (16GB or more recommended).
3. Web camera compatible with Raspberry Pi (Will be using USB Webcam for this experiment).
4. Internet connectivity (Wi-Fi).
5. Basic knowledge of Python and Linux commands.

---

**1. Introduction (10 minutes)**
Computer vision has been a very popular field since the advent of digital systems. However computer vision on the edge devices such as Raspberry Pi is challenging due to resource contraints. Edge Computer Vision (ECV) has emerged as a transformative technology, with [Gartner](https://www.linkedin.com/pulse/what-edge-computer-vision-how-get-started-deep-block-net) recognizing it as one of the top emerging technologies of 2023. ECV offers several benefits such as 1) they can operate in real-time or near-real-time, providing instant insights and enabling immediate actions, 2) they offer enhanced privacy and security and 3) It reduces dependency on network connectivity or relaxes the bandwidth requirements as some processing will be done within. 
In this lab, few basic and advanced image processing tasks on edge devices is introduced. An overview of the experiments/setup is as follows:
![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/882c84dc-1989-4039-807d-554a079e3776)

**2. Setting up the Raspberry Pi (15 minutes)**
- Booting up the Raspberry Pi.
- Setting up Wi-Fi/Ethernet.
- System updates:
  ```bash
  sudo apt update
  sudo apt upgrade
  ```
- **[Important!] Set up and activate a virtual environment named "image" for this experiment (to avoid conflicts in libraries) as below**
  ```bash
  sudo apt install python3-venv
  python3 -m venv image
  source image/bin/activate

**3. Connecting and Testing the Web Camera (5 minutes)**
- Physically connect the web camera to the Raspberry Pi.
  
**. Introduction to Real-time Image Processing with Python (25 minutes)**
- Installing OpenCV:
  ```bash
  pip install opencv-python  
  ```
- The [sample code](Codes/image_capture_display.py) shows the code to read frames from a webcam and then based on the intensity range for each colour channel (RGB), how to segment the image into red green and blue images. A sample image and the colour segmentation is as shown below:
  ![image](https://github.com/drfuzzi/INF2009_ImageAnalytics/assets/52023898/fd7c115d-0301-0d2-b2c1-7966dce3fec)
- Expand the code to segment another colour (say yellow)

**5. Real-time Image Analysis (25 minutes)**
- Installing scikit-image:
  ```bash
  pip install scikit-image  
  ```
- Computer vision employs feature extraction from images. Some important image features include edges and textures. In this section we will employ a feature named histogram of gradients (HoG) which is widely employed for face recognition and other tasks. HoG involves gradient operation (basically extracting edges) on various image patches (by dividing the image into blocks). A [sample code](Codes/image_hog_feature.py) involving scikit-image is employed for the same. The code displays the dominant HoG image for each image patch overlaid on the actual image. It has to be noted that OpenCV can also be employed for the same task, but the visualization using scikit-image is better compared to that from OpenCV. A sample image for the HoG feature is as shown below:
![image](https://github.com/drfuzzi/INF2009_ImageAnalytics/assets/52023898/94e7d597-c259-4634-a3dc-433c79e8533b)
  -  Note the usage of colour (RGB) to gray scale converion employed before HoG feature extraction.
  - Run the code with and without resizing the image and observe the resultant frame rate. It is important to note that for edge computing, downsizing the image will speed up the compute and many such informed decisions are critical.
  - Change the patch size in line 25 (feature.hog) and observe the changes in the results.
- The HoG features can be employed to identify the presence of face. An [example using OpenCV](Codes/image_human_capture.py) is available for experimenting with. A multiscale HoG feature extraction is employed in this case. This involves extracting HoG features at multiple scales (resolutions) of the given image. 

**6. Real-time Image Feature Analysis for Face Capture and Facial Landmark Extraction (20 minutes)**
- In this work, a light weight opensource library named *"Mediapipe"* for tasks such as face landmark detection, pose estimation, hand landmark detection, hand gesture recognition and object detection using pretrained neural network models.
- [MediaPipe](https://developers.google.com/mediapipe) is a on-device (*embedded machine learning*) framework for building cross platform multimodal applied ML pipelines that consist of fast ML inference, classic computer vision, and media processing (e.g. video decoding). MediaPipe was open sourced at CVPR in June 2019 as v0.5.0 and has various lightweight models developed with Tensorflow lite available for usage.
- Installing media pipe:
  ```bash  
  pip install mediapipe
  ```
- Try the [sample code](Codes/image_face_capture.py) to detect the face based on Mediapipe's approach which is very light weight when compared to the approach employed in above section. Observe the speed up. - A sample image with face landmarks is as shown below:
![Mediapipe Face Mesh_screenshot_18 01 2025](https://github.com/user-attachments/assets/3e952cbb-72df-4258-9d96-83f05c741096)

- [Optional] An opencv alternative (no dependence on mediapipe) of the face detection is available in the [sample code](Codes/image_human_capture_opencv.py). If you are using this code, make sure you download the [Haar cascade model](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml) manually and save it as 'haarcascade_frontalface_alt2.xml' in the same folder as the code. 
---

**[Optional] Homework/Extended Activities:**
1. Explore more advanced OpenCV functionalities like SIFT, SURF, and ORB for feature detection. These features alongside HoG could be used for image matching (e.g. face recognition)
2. Build an eye blink detection system for drowsiness detection.  

---

**Resources:**
1. Raspberry Pi official documentation.
2. OpenCV documentation and tutorials.
3. Relevant Python libraries documentation for image processing (e.g., `opencv`, `scikit-image`, `mediapipe`).

---

