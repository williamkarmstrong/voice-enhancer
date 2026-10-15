import numpy as np

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