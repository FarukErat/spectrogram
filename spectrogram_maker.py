import librosa
import numpy as np
from PIL import Image

def draw_spectrogram(audio_file_path, image_file_path,
                     min_hz=None, max_hz=None, min_db=None, max_db=None):
    data, sample_rate = librosa.load(audio_file_path, sr=None)
    amplitude_array = librosa.stft(data)
    spectrogram, _ = librosa.magphase(amplitude_array)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    frequency_array = librosa.fft_frequencies(sr=sample_rate)

    if min_hz is None:
        min_hz = frequency_array[0]
    if max_hz is None:
        max_hz = frequency_array[-1]

    if min_db is None:
        min_db = np.min(spectrogram_db)
    if max_db is None:
        max_db = np.max(spectrogram_db)

    frequency_mask = (frequency_array >= min_hz) & (frequency_array <= max_hz)
    spectrogram_db_filtered = spectrogram_db[frequency_mask, :]
    spectrogram_db_filtered = np.clip(spectrogram_db_filtered, min_db, max_db)

    # Normalize the spectrogram to the range [0, 255] for image representation
    spectrogram_db_normalized = 255 * (spectrogram_db_filtered - min_db) / (max_db - min_db)
    spectrogram_db_normalized = spectrogram_db_normalized.astype(np.uint8)

    # Flip the array vertically to match image coordinate system
    spectrogram_db_flipped = np.flipud(spectrogram_db_normalized)

    image = Image.fromarray(spectrogram_db_flipped)
    image.save(image_file_path)

draw_spectrogram('record.wav', 'spectrogram.png', min_db=-35, max_hz=2500)
