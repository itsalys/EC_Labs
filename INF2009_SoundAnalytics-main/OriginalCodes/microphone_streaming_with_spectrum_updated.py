#%% Import the required libraries
import sounddevice as sd # Refer to https://python-sounddevice.readthedocs.io/en/0.4.6/
import numpy as np 
import matplotlib.pyplot as plt
import time

#%% Parameters
BUFFER = 1024 * 16           # samples per frame (you can change the same to acquire more or less samples)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
RECORD_SECONDS = 30          # Specify the time to record from the microphone in seconds

#%% create matplotlib figure and axes with initial random plots as placeholder
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))
# create a line object with random data
x = np.arange(0, 2*BUFFER, 2)       # samples (waveform)
xf = np.fft.fftfreq(BUFFER,1/RATE)[:BUFFER//2]
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

#%% Reconrding the sound and constructing the spectrum
exec_time = []
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):   
       
    # Record the sound in int16 format and wait till recording is done
    data = sd.rec(frames=BUFFER,samplerate=RATE,channels=CHANNELS,dtype='int16',blocking=True)
    data = np.squeeze(data)  
    
    # compute FFT    
    start_time=time.time()  # for measuring frame rate
    fft_data = np.fft.fft(data)
    fft_data = np.abs(fft_data[:BUFFER//2])
    
    # calculate time of execution of FFT
    exec_time.append(time.time() - start_time)
    
    #update line plots for both axes
    line.set_ydata(data)
    line_fft.set_ydata(fft_data)
    line_fft.set_ydata(2.0/BUFFER * fft_data)
    fig.canvas.draw()
    fig.canvas.flush_events()

  
print('stream stopped')
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time)*1000))