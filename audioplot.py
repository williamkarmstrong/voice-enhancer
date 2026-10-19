import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
SAVE_PLOTS = False
CLOSE_AUDIO = os.path.join(HERE, "original_speech_5cm.wav")
FAR_AUDIO = os.path.join(HERE, "original_speech_1m.wav")
CONSONANTS_HZ = (3000, 20000)
NOISE_HZ = (20, 70)

# Held vowels found by listening (word, start s, end s, f0 Hz)
CLOSE_VOWELS = [("welcome", 0.61, 0.71, 108), ("four", 2.55, 2.70, 83),
                ("shreyas", 3.89, 4.04, 96), ("one", 5.72, 5.92, 82),
                ("transforms", 7.16, 7.24, 75)]
FAR_VOWELS = [("welcome", 1.19, 1.29, 95), ("four", 3.25, 3.40, 87),
              ("shreyas", 4.52, 4.66, 89), ("one", 6.05, 6.20, 84),
              ("transforms", 7.56, 7.68, 79)]


def save_svg(fig, filename):
    if SAVE_PLOTS:
        os.makedirs(PLOTS, exist_ok=True)
        fig.savefig(os.path.join(PLOTS, filename + ".svg"))


def import_wav_audio(filename):
    """Load a 16-bit .wav scaled to -1 to 1, averaging stereo to mono."""
    fs, x = wavfile.read(filename)
    x = x / 32768
    if x.ndim > 1:
        x = x.mean(axis=1)
    return fs, x


def spectrum_db(x, fs):
    """Positive-frequency bins in Hz and their amplitudes in dB."""
    n = len(x)
    bins = np.arange(1, n // 2)
    freqs = bins * fs / n
    # Divide by n so the level doesn't depend on recording length
    return freqs, 20 * np.log10(np.abs(np.fft.fft(x)[bins]) / n)


def f0_range(vowels):
    f0s = [f0 for _, _, _, f0 in vowels]
    return min(f0s), max(f0s)


def shade_bands(ax, f0_hz, consonants_hz, noise_hz):
    """Mark the f0, harmonic, consonant and noise bands, each given as (low, high) Hz."""
    low, high = f0_hz
    ax.axvspan(low, high, color="tab:green", alpha=0.3,
               label=f"Vowel $f_0$ ({low}–{high} Hz)")

    # Harmonic n spans n*low to n*high, until the bands start to overlap
    last = int(high / (high - low))
    for n in range(2, last + 1):
        label = f"Harmonics $2f_0$–${last}f_0$" if n == 2 else None
        ax.axvspan(n * low, n * high, color="tab:green", alpha=0.15, label=label)

    ax.axvspan(*consonants_hz, color="tab:orange", alpha=0.15, label="Consonants")
    ax.axvspan(*noise_hz, color="tab:red", alpha=0.15, label="Noise")


def plot_time_and_spectrum(x, fs, filename, f0_hz, consonants_hz, noise_hz):
    t = np.arange(len(x)) / fs
    freqs, db = spectrum_db(x, fs)

    fig, (time_ax, spec_ax) = plt.subplots(2, 1, figsize=(6.3, 4))

    time_ax.plot(t, x, linewidth=0.5)
    time_ax.set_xlabel("Time (s)")
    time_ax.set_ylabel("Normalised amplitude")
    time_ax.set_xlim(0, t[-1])
    time_ax.grid(alpha=0.3)

    spec_ax.semilogx(freqs, db, linewidth=0.5)
    spec_ax.set_xlabel("Frequency (Hz)")
    spec_ax.set_ylabel("Amplitude (dB)")
    spec_ax.set_xlim(20, fs / 2)  # 20 Hz (lowest audible) to Nyquist
    spec_ax.grid(which="both", alpha=0.3)
    shade_bands(spec_ax, f0_hz, consonants_hz, noise_hz)
    spec_ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    save_svg(fig, filename)

# Plot the spectrum of one vowel up to 1 kHz, marking f0 and its harmonics
def plot_vowel_spectrum(ax, x, fs, start, end, f0):
    segment = x[int(start * fs):int(end * fs)]
    # Hann window stops the slice edges smearing the peaks
    n = len(segment)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    freqs, db = spectrum_db(segment * window, fs)
    below_1khz = freqs < 1000

    ax.plot(freqs[below_1khz], db[below_1khz], linewidth=0.8, label="Spectrum")
    ax.axvline(f0, color="black", linestyle="--", linewidth=0.8,
               label=f"$f_0$ ≈ {f0} Hz")
    for harmonic_hz in range(2 * f0, 1000, f0):
        ax.axvline(harmonic_hz, color="black", linestyle=":", linewidth=0.8)
    ax.plot([], [], color="black", linestyle=":", linewidth=0.8,
            label="Harmonics ($n f_0$)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xlim(0, 1000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def save_vowel_spectrum(x, fs, start, end, filename, f0):
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    plot_vowel_spectrum(ax, x, fs, start, end, f0)
    fig.tight_layout()
    save_svg(fig, filename)
    plt.close(fig)


def plot_vowel_grid(x, fs, vowels, label, filename):
    fig, axes = plt.subplots(len(vowels), 1, figsize=(6.3, 10), sharex=True)
    fig.suptitle(f"{label}: spectra of the held vowels")

    for ax, (word, start, end, f0) in zip(axes, vowels):
        plot_vowel_spectrum(ax, x, fs, start, end, f0)
        ax.set_title(f'"{word}" ({start:.2f}–{end:.2f} s)', fontsize=10)
        ax.label_outer()

    fig.tight_layout()
    save_svg(fig, filename)


def analyse(path, vowels, name, label):
    fs, x = import_wav_audio(path)
    plot_time_and_spectrum(x, fs, f"{name}_audio", f0_range(vowels), CONSONANTS_HZ, NOISE_HZ)
    plot_vowel_grid(x, fs, vowels, label, f"{name}_vowels")

    if SAVE_PLOTS:
        for word, start, end, f0 in vowels:
            save_vowel_spectrum(x, fs, start, end, f"{name}_{word}", f0)


if __name__ == "__main__":
    analyse(CLOSE_AUDIO, CLOSE_VOWELS, "close", "Close mic")
    analyse(FAR_AUDIO, FAR_VOWELS, "far", "Far mic")
    plt.show()
