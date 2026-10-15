import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

CLOSE_AUDIO = "./audio/audio1_close.wav"
FAR_AUDIO = "./audio/audio2_far.wav"


# Imports a .wav file and normalises the 16-bit samples to the range
# -1 to 1. Stereo recordings are averaged down to a single channel.
def import_wav_audio(filename):
    fs, x = wavfile.read(filename)
    x = x / 32768
    if x.ndim > 1:
        x = x.mean(axis=1)
    return fs, x


# Plots the signal in the time domain and its frequency spectrum
# (dB vs log Hz) as two subplots, then saves them as an SVG.
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

    # Size for latex report
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 5))

    # Time domain: normalised amplitude vs seconds
    ax[0].plot(t, x, linewidth=0.5)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Normalised amplitude")
    ax[0].set_xlim(0, t[-1])
    ax[0].grid(alpha=0.3)

    # Frequency domain: dB vs log frequency, 20 Hz (lowest audible) to Nyquist
    ax[1].semilogx(F, xf_db, linewidth=0.5)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude (dB)")
    ax[1].set_xlim(20, fs / 2)
    ax[1].grid(which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("./plots/" + filename + ".svg")


# Loads both recordings, plots each one, then displays all plots
if __name__ == "__main__":
    fs, x = import_wav_audio(CLOSE_AUDIO)
    plot_time_vs_fft(x, fs, "close_audio")

    fs, x = import_wav_audio(FAR_AUDIO)
    plot_time_vs_fft(x, fs, "far_audio")

    plt.show()
