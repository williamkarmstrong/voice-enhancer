import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

CLOSE_AUDIO = "./audio/audio1_close.wav"
FAR_AUDIO = "./audio/audio2_far.wav"

"""
Imports a .wav audio file and normalises
a 16 bit integer to a value from -1 to 1.
Checks whether the file is stereo / mono
and if stereo averages the sample.
"""
def import_wav_audio(filename):
    fs, x = wavfile.read(filename)
    x = x / 32768
    if x.ndim > 1:
        x = x.mean(axis=1)
    return fs, x


"""
Plots the signal in the time domain and its
frequency spectrum (dB vs log Hz) as two
subplots, then saves them as a vector PDF.
"""
def plot_time_vs_fft(x, fs, filename):
    N = len(x)
    # Sample number to time in seconds
    t = np.arange(N) / fs

    # Bins 1 to N/2
    k = np.arange(1, N // 2)
    # Bin number -> frequency in Hz
    F = k * fs / N
    xf = np.fft.fft(x)
    # Magnitude in dB, scaled by N so it doesn't depend on recording length
    xf_db = 20 * np.log10(np.abs(xf[k]) / N)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    # Time domain: normalised amplitude vs seconds
    ax[0].plot(t, x, linewidth=0.5)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Normalised amplitude")

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
    plot_time_vs_fft(x, fs, "close_audio")

    fs, x = import_wav_audio(FAR_AUDIO)
    plot_time_vs_fft(x, fs, "far_audio")

    plt.show()
