#%% Import the required libraries
import pyaudio # Refer to https://people.csail.mit.edu/hubert/pyaudio/
import struct  # Refer to https://docs.python.org/3/library/struct.html (Used for converting audio read as bytes to int16)
import numpy as np 
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq # Refer to https://docs.scipy.org/doc/scipy/tutorial/fft.html (Used for Fourier Spectrum to display audio frequencies)
import time # In case time of execution is required

#%% Parameters
BUFFER = 1024 * 16           # samples per frame (you can change the same to acquire more or less samples)
FORMAT = pyaudio.paInt16     # audio format (bytes per sample)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
RECORD_SECONDS = 30          # Specify the time to record from the microphone in seconds

#%% create matplotlib figure and axes with initial random plots as placeholder
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))
# create a line object with random data
x = np.arange(0, 2*BUFFER, 2)       # samples (waveform)
xf = fftfreq(BUFFER, (1/RATE))[:BUFFER//2]

line, = ax1.plot(x,np.random.rand(BUFFER), '-', lw=2)
line_fft, = ax2.plot(xf,np.random.rand(BUFFER//2), '-', lw=2)

# basic formatting for the axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-5000, 5000) # change this to see more amplitude values (when we speak)
ax1.set_xlim(0, BUFFER)

ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000) 
ax2.set_xlim(0, RATE/2)

# Do not show the plot yet
plt.show(block=False)

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
    
    # compute FFT    
    start_time=time.time()  # for measuring frame rate
    yf = fft(data_int)
    
    # calculate time of execution of FFT
    exec_time.append(time.time() - start_time)
    
    #update line plots for both axes
    line.set_ydata(data_int)
    line_fft.set_ydata(2.0/BUFFER * np.abs(yf[0:BUFFER//2]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    
audio.terminate()
   
print('stream stopped')
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time)*1000))