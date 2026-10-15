import numpy as np
from scipy.ndimage import median_filter, minimum_filter1d
from loader import smooth_step, bell


def band(mags, f, low, high):
    return np.sum(mags[:, (f >= low) & (f <= high)] ** 2, axis=1)

def eq_gains(f, bass, mud, presence, air):
    gain = bass * bell(f, 150, 60) + mud * bell(f, 250, 80)
    gain = gain + presence * bell(f, 3000, 1000) + air * bell(f, 8000, 2000)
    return 10 ** (gain/20)

def pop_gains(mags, f):
    ratio = band(mags, f, 20, 200) / (band(mags, f, 300, 3000) + 1e-9)
    cut = np.where(ratio > 12, np.sqrt(0.3 / ratio), 1)
    return np.where(f < 1000, minimum_filter1d(cut, 7)[:, None], 1)

def click_gains(mags, f):
    local = median_filter(mags, size=(21, 1))
    loud = np.sum(mags ** 2, axis=1) > 10 * np.sum(local ** 2, axis=1)
    loud = np.convolve(loud, [1, 1, 1], mode="same") > 0
    return np.where(loud[:, None], np.minimum(1, local / mags + 1e-9), 1)

def sibilance_gains(mags, f):
    hissy = band(mags, f, 4000, 12000) > 0.25 * np.sum(mags ** 2, axis=1)
    return np.where(hissy[:, None] & (f >= 2500), 0.2, 1)