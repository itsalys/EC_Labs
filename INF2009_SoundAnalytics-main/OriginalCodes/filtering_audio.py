#%% Import the required libraries
import pyaudio # Refer to https://people.csail.mit.edu/hubert/pyaudio/
import struct  # Refer to https://docs.python.org/3/library/struct.html (Used for converting audio read as bytes to int16)
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt # Refer to https://docs.scipy.org/doc/scipy/reference/signal.html (Used for Bandpass filtering)
import time # In case time of execution is required  

#%% Parameters
BUFFER = 1024 * 16          # samples per frame (you can change the same to acquire more or less samples)
FORMAT = pyaudio.paInt16    # audio format (bytes per sample)
CHANNELS = 1                # single channel for microphone
RATE = 44100                # samples per second
RECORD_SECONDS = 20         # Specify the time to record from the microphone in seconds

#%% create matplotlib figure and axes with initial random plots as placeholder
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))
# create a line object with random data
x = np.arange(0, 2*BUFFER, 2)       # samples (waveform)

line, = ax1.plot(x,np.random.rand(BUFFER), '-', lw=2)
line_filter, = ax2.plot(x,np.random.rand(BUFFER), '-', lw=2)

# basic formatting for the axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('amplitude')
ax1.set_ylim(-5000, 5000) # change this to see more amplitude values (when we speak)
ax1.set_xlim(0, BUFFER)

ax2.set_title('FILTERED')
ax2.set_xlabel('samples')
ax2.set_ylabel('amplitude')
ax2.set_ylim(-5000, 5000) 
ax2.set_xlim(0, BUFFER)

# show the plot
plt.show(block=False)

#%% Function for design of filter
def design_filter(lowfreq, highfreq, fs, order=3):
    nyq = 0.5*fs
    low = lowfreq/nyq
    high = highfreq/nyq
    sos = butter(order, [low,high], btype='band',output='sos')
    return sos

# design the filter
sos = design_filter(19400, 19600, 48000, 3) #change the lower and higher freqcies according to choice


#%% Initialize the pyaudio class instance
audio = pyaudio.PyAudio()

# stream object to get data from microphone
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=BUFFER
)

print('stream started')
exec_time = []
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):   
       
    # binary data
    data = stream.read(BUFFER)  
   
    # convert data to 16bit integers
    data_int = struct.unpack(str(BUFFER) + 'h', data)    
    
    # Bandpass filtering
    start_time=time.time()  # for measuring frame rate
    yf = sosfilt(sos, data_int)
    
    # calculate average frame rate
    exec_time.append(time.time() - start_time)
    
    #update line plots for both axes
    line.set_ydata(data_int)
    line_filter.set_ydata(yf)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
audio.terminate()

print('stream stopped')
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time)*1000))