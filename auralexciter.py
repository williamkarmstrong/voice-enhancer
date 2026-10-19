import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from audioplot import import_wav_audio, spectrum_db, save_svg, HERE
from audioenhancer import enhance

CUT_HZ = 2000    # only content below this is distorted
KEEP_HZ = 4000   # harmonics below this are removed
TOP_HZ = 10000   # harmonics above this are removed
DRIVE = 5        # distortion amount
MIX = 0.1        # level of the harmonics added to the original

# (drive, mix) pairs compared in the iterations plot
ITERATIONS = [(5, 0.1), (5, 0.05), (2, 0.1)]

CLOSE_WAV = os.path.join(HERE, "enhanced_speech_5cm.wav")
FAR_WAV = os.path.join(HERE, "enhanced_speech_1m.wav")


def normalise(x):
    return x / np.max(np.abs(x))


def excite(x, fs, drive=DRIVE, mix=MIX):
    low_band = enhance(x, fs, [(0, CUT_HZ, 0)])
    harmonics = np.tanh(drive * normalise(low_band))
    high_harmonics = enhance(harmonics, fs, [(0, KEEP_HZ, 0), (TOP_HZ, fs / 2, 0)])
    return x + mix * normalise(high_harmonics)


def save_wav(y, fs, filename):
    # Clip to the int16 range so the conversion cannot overflow
    y = np.clip(y, -1, 32767 / 32768)
    wavfile.write(os.path.join(HERE, filename), fs, (y * 32768).astype(np.int16))


def plot_spectra(x, y, fs, label, filename):
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5), sharex=True, sharey=True)
    fig.suptitle(f"{label}: spectrum before and after exciter")

    for ax, signal, title in [(axes[0], x, "Before exciter"), (axes[1], y, "After exciter")]:
        freqs, db = spectrum_db(signal, fs)
        ax.semilogx(freqs, db, linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Amplitude (dB)")
        ax.set_xlim(20, fs / 2)
        ax.grid(which="both", alpha=0.3)
        for hz, name in [(CUT_HZ, "CUT"), (KEEP_HZ, "KEEP"), (TOP_HZ, "TOP")]:
            ax.axvline(hz, color="grey", linestyle="--")
            ax.text(hz, -60, f" {name}", fontsize=8)

    axes[1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    save_svg(fig, filename)


# Sum the energy in each frequency band
def band_energy(signal, fs, band_edges):
    edge_bins = (band_edges * len(signal) / fs).astype(int)
    power = np.abs(np.fft.fft(signal)) ** 2

    energies = []
    for start, end in zip(edge_bins, edge_bins[1:]):
        energies.append(power[start:end].sum())
    return np.array(energies)


def plot_iterations(x, fs):
    # 13 third-octave bands starting at 1 kHz
    band_edges = 1000 * 2 ** (np.arange(14) / 3)
    band_centres = np.sqrt(band_edges[:-1] * band_edges[1:])
    energy_before = band_energy(x, fs, band_edges)

    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.set_title("Close mic: level added by the exciter")

    for drive, mix in ITERATIONS:
        y = excite(x, fs, drive, mix)
        energy_after = band_energy(y, fs, band_edges)
        added_db = 10 * np.log10(energy_after / energy_before)
        ax.semilogx(band_centres, added_db, marker="o", markersize=3,
                    label=f"DRIVE {drive}, MIX {mix}")

    arrow = dict(arrowstyle="->")
    ax.annotate("MIX 0.05: ~2 dB less", xy=(5657, 2.2), xytext=(2000, 10), arrowprops=arrow)
    ax.annotate("DRIVE 2: ~10 dB less", xy=(8980, 17), xytext=(2000, 18), arrowprops=arrow)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level added (dB)")
    ax.set_xlim(1000, 20000)
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_svg(fig, "exciter_iterations")


def process(path, label, plot_name, wav_name=None):
    fs, x = import_wav_audio(path)
    y = excite(x, fs)
    plot_spectra(x, y, fs, label, plot_name)
    if wav_name:
        save_wav(y, fs, wav_name)


if __name__ == "__main__":
    process(CLOSE_WAV, "Close mic", "close_exciter", "exciter_speech.wav")
    process(FAR_WAV, "Far mic", "far_exciter")

    fs, x = import_wav_audio(CLOSE_WAV)
    plot_iterations(x, fs)

    plt.show()
