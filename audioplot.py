import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Paths are relative to this file so it runs from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")

# Set True to write the report's SVGs to plots
SAVE_PLOTS = False
CLOSE_AUDIO = os.path.join(HERE, "original_speech_5cm.wav")
FAR_AUDIO = os.path.join(HERE, "original_speech_1m.wav")

# Held vowels found by listening: (word, start s, end s, f0 Hz)
CLOSE_VOWELS = [("welcome", 0.61, 0.71, 108), ("four", 2.55, 2.70, 83),
                ("shreyas", 3.89, 4.04, 96), ("one", 5.72, 5.92, 82),
                ("transforms", 7.16, 7.24, 75)]
FAR_VOWELS = [("welcome", 1.19, 1.29, 95), ("four", 3.25, 3.40, 87),
              ("shreyas", 4.52, 4.66, 89), ("one", 6.05, 6.20, 84),
              ("transforms", 7.56, 7.68, 79)]


# Writes a figure to plots/ as an SVG
def save_svg(fig, filename):
    if SAVE_PLOTS:
        os.makedirs(PLOTS, exist_ok=True)
        fig.savefig(os.path.join(PLOTS, filename + ".svg"))


# (min, max) f0 of a recording's vowels
def f0_band(vowels):
    f0s = [f0 for _, _, _, f0 in vowels]
    return min(f0s), max(f0s)


# Loads a 16-bit .wav scaled to -1 to 1, averaging stereo to mono
def import_wav_audio(filename):
    fs, x = wavfile.read(filename)
    x = x / 32768
    if x.ndim > 1:
        x = x.mean(axis=1)
    return fs, x


# Plots the time signal and its spectrum (dB vs log Hz)
# marking the f0, harmonic, consonant and noise bands ((low, high) Hz).
def plot_time_vs_fft(x, fs, filename, f0_band, consonants, noise):
    N = len(x)
    t = np.arange(N) / fs

    # Positive frequency bins and their frequencies in Hz
    k = np.arange(1, N // 2)
    F = k * fs / N
    xf = np.fft.fft(x)
    
    # Divide by N so the level doesn't depend on recording length
    xf_db = 20 * np.log10(np.abs(xf[k]) / N)

    # Fits the report's text width
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 4))

    ax[0].plot(t, x, linewidth=0.5)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Normalised amplitude")
    ax[0].set_xlim(0, t[-1])
    ax[0].grid(alpha=0.3)

    # 20 Hz (lowest audible) to Nyquist
    ax[1].semilogx(F, xf_db, linewidth=0.5)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude (dB)")
    ax[1].set_xlim(20, fs / 2)
    ax[1].grid(which="both", alpha=0.3)

    # Harmonic band n spans n*low to n*high
    low, high = f0_band
    ax[1].axvspan(low, high, color="tab:green", alpha=0.3,
                  label=f"Vowel $f_0$ ({low}–{high} Hz)")
    n_max = int(high / (high - low))
    for n in range(2, n_max + 1):
        ax[1].axvspan(n * low, n * high, color="tab:green", alpha=0.15,
                      label=f"Harmonics $2f_0$–${n_max}f_0$" if n == 2 else None)
    ax[1].axvspan(*consonants, color="tab:orange", alpha=0.15, label="Consonants")
    ax[1].axvspan(*noise, color="tab:red", alpha=0.15, label="Noise")
    ax[1].legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    save_svg(fig, filename)


# Draws the spectrum of one vowel slice with f0 and n*f0 marked
def draw_segment_fft(ax, x, fs, start, end, f0):
    segment = x[int(start * fs):int(end * fs)]
    N = len(segment)
    # Hann window stops the slice edges smearing the peaks
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)
    # Only bins up to 1 kHz
    k = np.arange(1, 1000 * N // fs)
    F = k * fs / N
    segment_db = 20 * np.log10(np.abs(np.fft.fft(segment * window)[k]) / N)

    ax.plot(F, segment_db, linewidth=0.8, label="Spectrum")
    ax.axvline(f0, color="black", linestyle="--", linewidth=0.8,
               label=f"$f_0$ ≈ {f0} Hz")
    ax.vlines(f0 * np.arange(2, 1000 // f0 + 1), 0, 1,
              transform=ax.get_xaxis_transform(), color="black",
              linestyle=":", linewidth=0.8, label="Harmonics ($n f_0$)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xlim(0, 1000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# Saves one vowel spectrum as an SVG
def plot_segment_fft(x, fs, start, end, filename, f0):
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    draw_segment_fft(ax, x, fs, start, end, f0)
    fig.tight_layout()
    save_svg(fig, filename)
    plt.close(fig)


# Shows every vowel of a recording in one titled figure
def plot_vowel_grid(x, fs, vowels, label, filename):
    fig, axes = plt.subplots(len(vowels), 1, figsize=(6.3, 10), sharex=True)
    fig.canvas.manager.set_window_title(f"{label} – vowel spectra")
    fig.suptitle(f"{label}: spectra of the held vowels")
    for ax, (word, start, end, f0) in zip(axes, vowels):
        draw_segment_fft(ax, x, fs, start, end, f0)
        ax.set_title(f'"{word}" ({start:.2f}–{end:.2f} s)', fontsize=10)
        ax.label_outer()

    fig.tight_layout()
    save_svg(fig, filename)


if __name__ == "__main__":
    fs, x = import_wav_audio(CLOSE_AUDIO)
    plot_time_vs_fft(x, fs, "close_audio", f0_band(CLOSE_VOWELS), (3000, 20000), (20, 70))
    plt.gcf().canvas.manager.set_window_title("Close mic – time signal and spectrum")
    if SAVE_PLOTS:
        for word, start, end, f0 in CLOSE_VOWELS:
            plot_segment_fft(x, fs, start, end, "close_" + word, f0)
    plot_vowel_grid(x, fs, CLOSE_VOWELS, "Close mic", "close_vowels")

    fs, x = import_wav_audio(FAR_AUDIO)
    plot_time_vs_fft(x, fs, "far_audio", f0_band(FAR_VOWELS), (3000, 20000), (20, 70))
    plt.gcf().canvas.manager.set_window_title("Far mic – time signal and spectrum")
    if SAVE_PLOTS:
        for word, start, end, f0 in FAR_VOWELS:
            plot_segment_fft(x, fs, start, end, "far_" + word, f0)
    plot_vowel_grid(x, fs, FAR_VOWELS, "Far mic", "far_vowels")

    plt.show()
