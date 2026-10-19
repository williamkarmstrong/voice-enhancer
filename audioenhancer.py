import numpy as np


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