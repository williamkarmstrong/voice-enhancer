import numpy as np
from scipy.io import wavfile
from loader import import_wav_audio, process_frames, CLOSE_AUDIO, FAR_AUDIO
from noise_remover import noise_gains
from enhancer import eq_gains, pop_gains, click_gains, sibilance_gains


SETTINGS = {
    "close": dict(file=CLOSE_AUDIO, noise=3.0, close_mic=True, bass=0.0, mud=-2.5, presence=4.0, air=1.0),
    "far": dict(file=FAR_AUDIO, noise=2.0, close_mic=False, bass=5.0, mud=0.0, presence=5.0, air=0.0),
}

def enhance_audio(x, fs, s):
    def gains(mags, f):
        g = noise_gains(mags, f, s["noise"])
        g *= eq_gains(f, s["bass"], s["mud"], s["presence"], s["air"])
        if s["close_mic"]:
            g *= pop_gains(mags, f)
            g *= click_gains(mags, f)
            g *= sibilance_gains(mags, f)
        return g
    return process_frames(x - x.mean(), fs, gains)

if __name__ == "__main__":
    for name, s in SETTINGS.items():
        fs, x = import_wav_audio(s["file"])
        y = enhance_audio(x.astype(np.float64), fs, s)
        y = y/ np.max(np.abs(y)) * 0.9
        wavfile.write(f"./audio/{name}_enhanced.wav", fs, (y * 32767).astype(np.int16))
        print(f"Enhanced audio saved to ./audio/{name}_enhanced.wav")