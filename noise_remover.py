import numpy as np
from scipy.io import wavfile
from loader import import_wav_audio


def smooth_step(f, f0, width):
    return 0.5 * (1 + np.tanh((f - f0) / width))

def bell(f, centre, width):
    return np.exp(-((f - centre) ** 2) / (2 * width ** 2))

def close_gain(f):
    db = np.zeros_like(f)
    db += -2.5*bell(f, 220, 80)
    db += 4.0*bell(f, 3500, 1200)
    db += 2.0*smooth_step(f, 9000, 1500)
    g = 10 **(db / 20)
    g *= smooth_step(f, 80, 15)
    g *= 1 - 0.9 * smooth_step(f, 14000, 1500)
    return g


def remove_pops(x, fs, frame=1024, threshold=12.0, target=0.5, hold=2):
    hop = frame // 2
    window = np.sqrt(np.hanning(frame + 1)[:-1])
    f = np.fft.rfftfreq(frame, 1/fs)
    low = (f >= 20) & (f < 200)
    mid = (f >= 300) & (f < 3000)
    low_shape = 1 - smooth_step(f, 250, 30)

    padded = np.concatenate([np.zeros(hop), x, np.zeros(frame)])
    starts = range(0, len(padded) - frame, hop)
    spectra = [np.fft.rfft(padded[i:i + frame] * window) for i in starts]
    ratios = np.array([np.sum(np.abs(F[low]) ** 2) / (np.sum(np.abs(F[mid]) ** 2) + 1e-9)
                       for F in spectra])

    # A pop lasts a few frames, so keep cutting for `hold` frames either side.
    cuts = np.ones(len(ratios))
    for k in np.where(ratios > threshold)[0]:
        cut = np.sqrt(target / ratios[k])
        lo_k, hi_k = max(0, k - hold), min(len(cuts), k + hold + 1)
        cuts[lo_k:hi_k] = np.minimum(cuts[lo_k:hi_k], cut)

    y = np.zeros_like(padded)
    for i, F, cut in zip(starts, spectra, cuts):
        F = F * (1 - (1 - cut) * low_shape)
        y[i:i + frame] += np.fft.irfft(F, frame) * window
    return y[hop:hop + len(x)]


def enhance_close(input_audio, output_audio): 
    
    fs, x = import_wav_audio(input_audio)
    
    x = x.astype(np.float64)
    x = x - x.mean()
    x = remove_pops(x, fs)
    length = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(length, 1/fs)
    
    Y  = X* close_gain(f)
    y = np.fft.irfft(Y, length)
    
    y = y/np.max(np.abs(y)) * 0.9
    
    wavfile.write(output_audio, fs, (y* 32767).astype(np.int16))
    
    return fs, x, y


if __name__ == "__main__":
    enhance_close("./audio/audio1_close.wav", "./audio/audio1_close_enhanced.wav")
    
