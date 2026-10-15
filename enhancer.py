import numpy as np


def enhance(x, fs, bands):
    X = np.fft.fft(x)
    N = len(X)
    
    for low, high, gain in bands:
        idx_low = int((low * N) / fs)
        idx_high = int((high * N) / fs)
        X[idx_low:idx_high] *= gain
        if idx_low > 0:
            X[-idx_high:-idx_low] *= gain
        elif idx_low == 0 and idx_high > 0:
            X[-idx_high+1:] *= gain
            
    return np.fft.ifft(X).real