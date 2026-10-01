import numpy as np
from scipy.io import wavfile 

CLOSE_AUDIO = "/audio/audio1_close.wav"
FAR_AUDIO = "/audio/audio2_far.wav"

def import_audio(filename):
    fs, x = wavfile.read(filename)
    print(fs,x[:10])

import_audio(CLOSE_AUDIO)
import_audio(FAR_AUDIO)
