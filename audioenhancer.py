import numpy as np
from scipy.io import wavfile
from audioplot import import_wav_audio


def remove_noise(x, fs, strength=1.0, close_mic=False):
    X = np.fft.fft(x)
    N = len(X)
    half_N = N // 2
    
    low_cutoff = 100 if close_mic else 80
    high_cutoff = 10000 
    
    def freq_to_index(freq):
        return int((freq * N) / fs)
    
    if high_cutoff is not None:
        idx_high = freq_to_index(high_cutoff)
        X[idx_high:half_N] = 0
        X[half_N:-idx_high] = 0
        
    if low_cutoff is not None:
        idx_low = freq_to_index(low_cutoff)
        X[0:idx_low] = 0
        if idx_low > 0:
            X[-idx_low:] = 0
    return np.fft.ifft(X).real


def enhance(x, fs, bands):
    X = np.fft.fft(x)
    N = len(X)
    
    for low, high, gain in bands:
        idx_low = int((low * N) / fs)
        idx_high = int((high * N) / fs)
        X[idx_low:idx_high] *= gain
        if idx_low == 0:
            if idx_high > 1:
                X[-idx_high + 1:] *= gain
        else:
            X[-idx_high + 1 : -idx_low + 1] *= gain
            
    return np.fft.ifft(X).real


RECORDINGS = {
    "close": dict(file="audio1_close.wav", strength=1.0, close_mic=True, end=7.9,
                  bands=[
                      (0, 80, 0),         
                      (80, 150, 0.7),     
                      (150, 300, 1.15),    
                      (2500, 5000, 1.25),  
                      (8000, 22050, 0)     
                  ]),
    "far":   dict(file="audio2_far.wav", strength=2.0, close_mic=False, end=None,
                  bands=[
                      (0, 120, 0),        
                      (150, 300, 1.3),     
                      (400, 800, 0.8),  
                      (2500, 5500, 1.15),  
                      (6500, 22050, 0)     
                  ]),
}


if __name__ == "__main__":
    for name, r in RECORDINGS.items():
        fs, x = import_wav_audio("audio/" + r["file"])
        y = remove_noise(x - x.mean(), fs, r["strength"], close_mic=r["close_mic"])
        y = enhance(y, fs, r["bands"])
        if r["end"]:
            y[int(r["end"] * fs):] = 0
        y = np.clip(y / np.std(y) * 0.08, -0.9, 0.9)
        wavfile.write(f"audio/{name}_enhanced.wav", fs, (y * 32767).astype(np.int16))
