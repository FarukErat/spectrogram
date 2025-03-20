import librosa
import numpy as np
from PIL import Image

def draw_spectrogram(audio_file_path, image_file_path,
                     min_hz=None, max_hz=None, min_db=None, max_db=None):
    # Load the audio file
    y, sr = librosa.load(audio_file_path, sr=None)

    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)

    # Convert the complex-valued STFT to amplitude
    S, _ = librosa.magphase(D)

    # Convert amplitude to decibels
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Determine frequency range
    freqs = librosa.fft_frequencies(sr=sr)
    if min_hz is None:
        min_hz = freqs[0]
    if max_hz is None:
        max_hz = freqs[-1]

    # Determine decibel range
    if min_db is None:
        min_db = np.min(S_db)
    if max_db is None:
        max_db = np.max(S_db)

    # Apply frequency mask
    freq_mask = (freqs >= min_hz) & (freqs <= max_hz)
    S_db = S_db[freq_mask, :]

    # Clip decibel values to the specified range
    S_db = np.clip(S_db, min_db, max_db)

    # Normalize the spectrogram to the range [0, 255] for image representation
    S_db_normalized = 255 * (S_db - min_db) / (max_db - min_db)
    S_db_normalized = S_db_normalized.astype(np.uint8)

    # Flip the array vertically to match image coordinate system
    S_db_flipped = np.flipud(S_db_normalized)

    # Convert the normalized spectrogram to a PIL Image
    image = Image.fromarray(S_db_flipped)

    # Save the image
    image.save(image_file_path)

# Example usage:
draw_spectrogram('record.wav', 'spectrogram.png', min_db=-35, max_hz=2500)
