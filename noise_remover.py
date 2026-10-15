import numpy as np 

def smooth_step(f, f1, width):
    return 1 / (1 + np.exp(-(f - f1) / width))


def noise_gains(mags, f, strength):
    energy = np.sum(mags ** 2, axis=1)
    noise = mags[energy <= np.percentile(energy[energy > 0], 10)].mean(axis=0)
    gains = np.maximum(0.1, 1 - strength * noise / (mags + 1e-9))
    return gains * smooth_step(f, 80, 15) * (1 - 0.9 * smooth_step(f, 14000, 1500))
