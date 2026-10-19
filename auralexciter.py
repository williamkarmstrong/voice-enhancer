import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from audioplot import import_wav_audio, spectrum_db, save_svg, HERE
from audioenhancer import enhance

CUT, DRIVE, KEEP, TOP, MIX = 2000, 5, 4000, 10000, 0.1

CLOSE_ENHANCED = os.path.join(HERE, "enhanced_speech_5cm.wav")
FAR_ENHANCED = os.path.join(HERE, "enhanced_speech_1m.wav")

RECORDINGS = [(CLOSE_ENHANCED, "Close mic (enhanced)", "close_exciter", "exciter_speech.wav"),
            (FAR_ENHANCED, "Far mic (enhanced)", "far_exciter", None)]

def excite(x, fs):
    # highpass
    side = enhance(x, fs, [(0, CUT, 0)])

    # make harmonics
    g = np.tanh(DRIVE * side / np.max(np.abs(side)))

    # keep the high ones, but not above TOP
    h = enhance(g, fs, [(0, KEEP, 0), (TOP, fs / 2, 0)])

    # mix into dry voice
    return x + MIX * h / np.max(np.abs(h))

# Plots the spectra before and after the exciter (dB vs log Hz)
def plot(x, y, fs, label, filename):
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 5), sharex=True, sharey=True)
    fig.canvas.manager.set_window_title(f"{label} – aural exciter spectrum")
    fig.suptitle(f"{label}: spectrum before and after exciter")

    for a, signal, title, colour in [(ax[0], x, "Before exciter", "tab:blue"),
                                     (ax[1], y, "After exciter", "tab:purple")]:
        F, db = spectrum_db(signal, fs)
        a.semilogx(F, db, linewidth=0.5, color=colour)
        a.set_title(title, fontsize=10)
        a.set_ylabel("Amplitude (dB)")
        a.set_xlim(20, fs / 2)
        a.grid(which="both", alpha=0.3)
        a.axvspan(20, CUT, color="tab:red", alpha=0.1,
                  label=f"Below CUT ({CUT} Hz): not excited")
        a.axvspan(KEEP, TOP, color="tab:green", alpha=0.15,
                  label=f"Above KEEP ({KEEP} Hz): harmonics added")
        a.axvspan(TOP, fs/2, color="tab:blue", alpha=0.15,
                          label=f"Above TOP ({TOP} Hz): harmonics zeroed")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[1].set_xlabel("Frequency (Hz)")

    fig.tight_layout()
    save_svg(fig, filename)


if __name__ == "__main__":
    for path, label, filename, out in RECORDINGS:
        fs, x = import_wav_audio(path)
        y = excite(x, fs)
        plot(x, y, fs, label, filename)
        if out:
            y = np.clip(y, -1, 32767 / 32768)
            wavfile.write(os.path.join(HERE, out), fs, (y * 32768).astype(np.int16))

    plt.show()
