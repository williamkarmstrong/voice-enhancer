import numpy as np
from scipy.io import wavfile
from audioplot import import_wav_audio, CLOSE_AUDIO, FAR_AUDIO
from noise_remover import remove_noise
from audioenhancer import enhance

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