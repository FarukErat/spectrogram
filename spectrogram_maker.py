import numpy as np
import scipy.io.wavfile as wav
from PIL import Image

def compute_spectrogram(signal, sample_rate, window_size=1024, hop_size=512):
    """
    Computes the spectrogram (magnitude of the FFT) of an audio signal.

    Parameters:
        signal (np.ndarray): 1D array of audio samples.
        sample_rate (int): Sampling rate of the audio signal.
        window_size (int): Number of samples per frame.
        hop_size (int): Number of samples to shift between frames.

    Returns:
        tuple: (spectrogram (2D np.ndarray), frequencies (1D np.ndarray))
            - spectrogram: 2D array (frequency bins x time frames) with magnitude values.
            - frequencies: Frequencies corresponding to each row in the spectrogram.
    """
    window = np.hamming(window_size)
    n_frames = 1 + (len(signal) - window_size) // hop_size
    spec_frames = []

    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start+window_size] * window
        # Compute FFT and take only the positive frequencies
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        spec_frames.append(magnitude)

    spectrogram = np.array(spec_frames).T
    # Compute frequency bins for a real FFT
    freqs = np.fft.rfftfreq(window_size, d=1./sample_rate)
    return spectrogram, freqs

def convert_to_db(spectrogram, eps=1e-10):
    """
    Converts the amplitude spectrogram to decibels (dB).

    Parameters:
        spectrogram (np.ndarray): 2D array of amplitude values.
        eps (float): Small constant to avoid log of zero.

    Returns:
        np.ndarray: Spectrogram in decibels.
    """
    return 20 * np.log10(spectrogram + eps)

def normalize_image(spec_db, min_db=None, max_db=None):
    """
    Normalizes the dB spectrogram to the range 0-255 for image conversion.

    Parameters:
        spec_db (np.ndarray): 2D array of spectrogram values in dB.
        min_db (float or None): Minimum dB value for scaling. If None, use spec_db.min().
        max_db (float or None): Maximum dB value for scaling. If None, use spec_db.max().

    Returns:
        np.ndarray: Normalized image array with type uint8.
    """
    # Determine scaling bounds
    if min_db is None:
        min_db = spec_db.min()
    if max_db is None:
        max_db = spec_db.max()

    # Clip the dB values
    spec_db_clipped = np.clip(spec_db, min_db, max_db)
    # Scale values to 0-255
    norm = (spec_db_clipped - min_db) / (max_db - min_db) * 255
    return norm.astype(np.uint8)

def save_spectrogram_image(signal, sample_rate, image_file_path,
                           window_size=1024, hop_size=512,
                           min_hz=None, max_hz=None,
                           min_db=None, max_db=None):
    """
    Computes the spectrogram of an audio signal, applies frequency and dB scaling,
    and saves it as an image.

    Parameters:
        signal (np.ndarray): 1D array of audio samples.
        sample_rate (int): Sampling rate of the audio signal.
        output_file (str): Filename for saving the spectrogram image.
        window_size (int): Number of samples per frame.
        hop_size (int): Number of samples to shift between frames.
        min_hz (float or None): Minimum frequency to display. If None, no lower bound is applied.
        max_hz (float or None): Maximum frequency to display. If None, no upper bound is applied.
        min_db (float or None): Minimum dB value for scaling. If None, use the minimum of the dB spectrogram.
        max_db (float or None): Maximum dB value for scaling. If None, use the maximum of the dB spectrogram.
    """
    # Compute the spectrogram and frequency bins
    spectrogram, freqs = compute_spectrogram(signal, sample_rate, window_size, hop_size)

    # Filter frequency bins if min_hz or max_hz are provided
    freq_indices = np.ones_like(freqs, dtype=bool)
    if min_hz is not None:
        freq_indices &= (freqs >= min_hz)
    if max_hz is not None:
        freq_indices &= (freqs <= max_hz)

    # Restrict the spectrogram and frequency bins to the specified range
    spectrogram = spectrogram[freq_indices, :]
    freqs = freqs[freq_indices]

    # Convert the amplitude spectrogram to dB
    spec_db = convert_to_db(spectrogram)

    # Normalize the spectrogram image using the specified dB range
    spec_image = normalize_image(spec_db, min_db, max_db)

    # Create and flip the image so that low frequencies appear at the bottom
    img = Image.fromarray(spec_image)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the image
    img.save(image_file_path)
    print(f"Spectrogram image saved to '{image_file_path}'.")

def process_wav_file(audio_file_path, image_file_path, window_size=1024, hop_size=512,
                     min_hz=None, max_hz=None, min_db=None, max_db=None):
    """
    Reads a WAV file, computes its spectrogram with optional frequency and dB scaling,
    and saves the spectrogram as an image.

    Parameters:
        filename (str): Path to the WAV file.
        output_file (str): Path where the output image will be saved.
        window_size (int): Number of samples per frame.
        hop_size (int): Number of samples to shift between frames.
        min_hz (float or None): Minimum frequency to display.
        max_hz (float or None): Maximum frequency to display.
        min_db (float or None): Minimum dB value for scaling.
        max_db (float or None): Maximum dB value for scaling.
    """
    sample_rate, data = wav.read(audio_file_path)

    # If stereo (or multi-channel), take only the first channel.
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize integer data to float if needed.
    if data.dtype not in [np.float32, np.float64]:
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val

    print(f"Processing '{audio_file_path}' (sample rate: {sample_rate} Hz)...")
    save_spectrogram_image(data, sample_rate, image_file_path, window_size, hop_size,
                           min_hz, max_hz, min_db, max_db)

if __name__ == "__main__":
    # Example usage:
    # Process the "record.wav" file and save the spectrogram image as "spectrogram.png"
    # Here you can specify frequency bounds and dB scaling if desired.
    process_wav_file("record.wav", "spectrogram.png",
                     min_hz=0, max_hz=2500,
                     min_db=-15, max_db=100)
