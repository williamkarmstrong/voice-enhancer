import numpy as np
from scipy.io import wavfile 
import matplotlib.pyplot as plt

CLOSE_AUDIO = "./audio/audio1_close.wav"
FAR_AUDIO = "./audio/audio2_far.wav"

def import_wav_audio(filename):
    fs, x = wavfile.read(filename)
    x = x.mean(axis=1)
    return fs, x

def plot_time_vs_fft(x, filename):
    xf = np.fft.fft(x)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(abs(x))
    ax[1].plot(abs(xf))
    fig.savefig("./plots/" + filename)


fs, x = import_wav_audio(CLOSE_AUDIO)
plot_time_vs_fft(x, "close_audio")

fs, x = import_wav_audio(FAR_AUDIO)
plot_time_vs_fft(x, "far_audio")
