import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from PIL import Image, ImageOps

def plot_spectrogram(audio_file_path, image_file_path,
                     nperseg=1024, noverlap=512,
                     invert=False, flip_y=True):
    # 1. Read
    fs, data = wavfile.read(audio_file_path)
    if data.ndim > 1:
        data = data[:, 0]

    # 2. Spectrogram
    f, t, Sxx = spectrogram(data, fs=fs,
                             window='hann',
                             nperseg=nperseg,
                             noverlap=noverlap,
                             scaling='spectrum')
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    # 3. Normalize
    S = Sxx_dB - Sxx_dB.min()
    S *= (255.0 / S.max())
    S = S.astype(np.uint8)

    # 4. Image
    img = Image.fromarray(S)
    if flip_y:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if invert:
        img = ImageOps.invert(img)

    img.save(image_file_path)
    print(f"Spectrogram saved to {image_file_path}")
