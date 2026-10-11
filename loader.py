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

def smooth_step(f, f1, width):
    return 1 / (1 + np.exp(-(f - f1) / width))

def bell(f, center, width):
    return np.exp(-((f - center) **2) / (2 * width **2))

def process_frames(x, fs, gain_rule, frame_size=1024 ):
    
    hop = frame_size //2
    window = np.sqrt(np.hanning(frame_size + 1)[:-1])
    padded = np.concatenate([np.zeros(hop), x, np.zeros(frame_size)])
    starts = range(0, len(padded) - frame_size, hop)
    
    spectrogram = np.array([np.fft.rfft(padded[i:i+frame_size] * window) for i in starts])
    f = np.fft.rfftfreq(frame_size, 1/fs)
    spectrogram = spectrogram * gain_rule(np.abs(spectrogram), f)
    
    y = np.zeros_like(padded)
    for i, F in zip(starts, spectrogram):
        y[i:i+frame_size] += np.fft.irfft(F) * window
    return y[hop:hop + len(x)]
    
if __name__ == "__main__":
    fs, x = import_wav_audio(CLOSE_AUDIO)
    plot_time_vs_fft(x, "close_audio")

    fs, x = import_wav_audio(FAR_AUDIO)
    plot_time_vs_fft(x, "far_audio")
