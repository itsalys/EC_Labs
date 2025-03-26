**Sound Analytics with Raspberry Pi 4/3 using Microphone**

**Objective:** By the end of this session, participants will understand how to set up a microphone with the Raspberry Pi, capture audio, and conduct basic sound analytics.

---

**Prerequisites:**
1. Raspberry Pi with Raspbian OS installed.
2. MicroSD card (16GB or more recommended).
3. USB keyboard, mouse, and monitor. You can also use Real VNC to access the Rasperry Pi (Need to enable VNC configuration in your Pi)
4. USB microphone compatible with Raspberry Pi.
5. (Optional) A Speaker/USB headset to hear the playback
6. Internet connectivity (Wi-Fi or Ethernet).
7. Basic knowledge of Python and Linux commands.

---

**1. Introduction (10 minutes)**
Developing a computer (ideally embedded) aided audio listening system similar to the human hearing to make sense of sounds is one of the growing areas of research. There are various application to the same not limited to 1) voice enabled services (e.g. Alexa Dot), 2) healthcare applications (e.g. lung sound analysis /  digital stethoscope) and 3) audio chatbots (e.g. IVR services). In this lab, we will be working on ways to capture audio data, analyze it and visualize the same on Raspberry Pi. A basic approach for the same is as shown below:
![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/bb9a2d8a-b4ae-4207-8272-21162987c821)

**2. Setting up the Raspberry Pi (10 minutes)**
- Booting up the Raspberry Pi.
- Setting up Wi-Fi/Ethernet.
- System updates:
  ```bash
  sudo apt update
  sudo apt upgrade
  ```
- **[Important!] Set up and activate a virtual environment named "audio" for this experiment (to avoid conflicts in libraries) as below**
  ```bash
  sudo apt install python3-venv
  python3 -m venv audio
  source audio/bin/activate

**3. Connecting and Testing the Microphone (15 minutes)**
- Physically connecting the microphone to the Raspberry Pi.
- Testing the microphone:
  ```bash
  arecord --duration=10 test.wav
  aplay test.wav
  ```
- Don't delete the .wav file as that will be used later for feature extraction

**4. Introduction to Sound Processing with Python (20 minutes)**
- Installing necessary Python libraries (may need additional libraries if not preinstalled with default python):
  ```bash
  sudo apt install portaudio19-dev
  pip3 install pyaudio
  pip3 install sounddevice
  pip3 install scipy matplotlib
  ```
  **Note that you need either pyaudio or sounddevice to record the audio stream from microphone**
- Capturing audio in Python.
- Fourier Transform: Understanding frequency components of sound.
- Visualizing sound waves (both the wave itself and the audio spectrum).
  - For above tasks, you can use the
     - [sample code](Codes/microphone_streaming_with_spectrum.py) if you are using *pyaudio*.
     - [sample code](Codes/microphone_streaming_with_spectrum_updated.py) if you are using *sounddevice*. 
  - A sample captured speech and its spectrum are shown below
    ![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/26449854-8770-46a7-ac2d-de94f8f2bc7a)
  - The top plot shows a sample time series of the captured audio and the bottom plot shows the frequency components present in the time series. It is easier to intepret audio by its spectrum.
  - Try speaking, making different sounds and observe how the spectrum changes.

    
**5. Basic Sound Analytics (40 minutes)**
- Filtering: Removing noise or specific frequencies. The below code illustrates a bandpass filter (only passes audio within a certain frequencies as decided by the user are kept).
- For the filtering task, you can use the
  - [sample code](Codes/filtering_audio.py) if you are using *pyaudio*.
  - [sample code](Codes/filtering_audio_updated.py) if you are using *sounddevice*.
  - Using the audio spectrum visualization, identify the frequency to be kept (e.g. tap sound or some particular sound) and change the above code accordingly.
- Feature extraction: Spectrogram, Chromogram, Mel-Spectrogram and MFFC.
  - Install the [librosa library](https://librosa.org/doc/latest/index.html) using the command
     ```bash
     pip install librosa
     ```
  - In this section, we will explore various features which can be extracted from speech/audio time series employing the librosa library. A [sample code](https://github.com/drfuzzi/INF2009_SoundAnalytics/blob/main/Codes/audio_features.py) which shows how to extract the above features is available for testing.
  - The time series recorded through microphone and saved as test.wav (mostly speech with some background noise) and its spectrogram are shown below \
    ![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/0ff75402-20c6-492f-9ee8-1f20c954c0a3) \
    The [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) is a visual representation of the spectrum of frequencies of a signal as it varies with time. \
  - The spectrogram and the chromogram are shown below \
    ![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/7be397f9-9b7e-4c98-a1ea-38dab4b2caba) \
    In Western music, the term [chroma feature or chromagram](https://en.wikipedia.org/wiki/Chroma_feature) closely relates to twelve different pitch classes. Chroma-based features, which are also referred to as "pitch class profiles", are a powerful tool for analyzing music whose pitches can be meaningfully categorized (often into twelve categories) and whose tuning approximates to the equal-tempered scale. 
  - The Mel-frequency spectrogram is shown below \
    ![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/4663a522-8c0e-416f-95e3-eefe42a3696b) \
    A Mel Spectrogram makes two important changes relative to a regular Spectrogram that plots Frequency vs Time. It uses the Mel Scale instead of Frequency on the y-axis. It uses the Decibel Scale instead of Amplitude to indicate colors. The [Mel Scale](https://en.wikipedia.org/wiki/Mel_scale) is a perceptual scale of pitches judged by listeners to be equal in distance from one another.
  - Finally the MFCC (Mel Frequency Cepstral Coefficients) and the original Mel Spectrogram are as shown below \
    ![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/d2746cc2-54a3-4eff-beb5-664813a2fcd0) \
    The [mel-frequency cepstrum (MFC)] (https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.

**6. Advanced Sound Analytics (20 minutes)**
- Introduction to machine learning with sound through speech recognition task (through [CMUSphinx](https://cmusphinx.github.io/wiki/) and [Google Speech Recognition](https://github.com/Uberi/speech_recognition/tree/master/third-party/Source%20code%20for%20Google%20API%20Client%20Library%20for%20Python%20and%20its%20dependencies).
- Installing the [speech_recognition library](https://github.com/Uberi/speech_recognition#readme) through following commands
  ```bash
  sudo apt-get install flac
  pip install pocketsphinx
  pip install SpeechRecognition
  ```
  - A FLAC encoder (installed through above command) is required to encode the audio data to send to the API. Similarly CM Sphinx is also installed through above commands
  - The [sample code](Codes/microphone_recognition.py) illustrates to record an audio, then use the CMUSphinx API and Google Speech Recognition APIs to predict the spoken text
  - Its important to see the 'Say Something' before you start speaking as in the initial few seconds the ambient noise is being captured.
  - A sample display screen will look like as in below:
    ![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/bc5b4ccc-f06e-422e-b0f0-8a403e14cc65)
    It is very clear from the screenshot that an offline (inference done on the edge device) model like Sphinx is not as effective as a Google Speech Recognition API where inference is done on the cloud. 
- Employ other speech recognition APIs provided in the [speech_recognition library](https://github.com/Uberi/speech_recognition#readme) and compare the performance on Rasperry Pi
- Modify the code to identify certain words in the generated (predicted text) which can form the basis for 'wake word' based system control (e.g. Ok Google, Alexa or Siri) 
---

**[Optional] Homework/Extended Activities:**
1. Build a voice-activated command system.
2. Create a basic sound classifier using a dataset of various sounds.
3. Experiment with sound effects: reverb, echo, and pitch alteration.

---

**Resources:**
1. Raspberry Pi official documentation.
2. Python sound processing library documentation (e.g., `librosa`, `pyaudio`, `speech_recognition`).
3. Online communities and forums related to Raspberry Pi and sound analytics.

---
