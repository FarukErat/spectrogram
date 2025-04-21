import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from PIL import Image, ImageOps

def plot_spectrogram(audio_file_path, image_file_path,
                     nperseg=1024, noverlap=512,
                     invert=False, flip_y=True):
    sample_rate, audio_data = wavfile.read(audio_file_path)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    sample_frequencies, segment_times, spectrogram_data = spectrogram(audio_data, fs=sample_rate,
                             window='hann',
                             nperseg=nperseg,
                             noverlap=noverlap,
                             scaling='spectrum')
    spectrogram_data_db = 10 * np.log10(spectrogram_data + 1e-10)

    S = spectrogram_data_db - spectrogram_data_db.min()
    S *= (255.0 / S.max())
    S = S.astype(np.uint8)

    img = Image.fromarray(S)
    if flip_y:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if invert:
        img = ImageOps.invert(img)

    img.save(image_file_path)
