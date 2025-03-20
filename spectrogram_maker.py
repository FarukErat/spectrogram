import numpy as np
import scipy.io.wavfile as wav
from PIL import Image

def compute_spectrogram(signal, window_size=1024, hop_size=512):
    """
    Computes the spectrogram (magnitude of the FFT) of an audio signal.

    Parameters:
        signal (np.ndarray): 1D array of audio samples.
        window_size (int): Number of samples per frame.
        hop_size (int): Number of samples to shift between frames.

    Returns:
        np.ndarray: 2D array of spectrogram values (frequency bins x time frames).
    """
    window = np.hamming(window_size)
    n_frames = 1 + (len(signal) - window_size) // hop_size
    spec_frames = []

    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start+window_size] * window
        # Compute the FFT and take the magnitude (only positive frequencies)
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        spec_frames.append(magnitude)

    # Convert list to a 2D numpy array and transpose so that
    # rows correspond to frequency bins and columns to time frames.
    spectrogram = np.array(spec_frames).T
    return spectrogram

def normalize_spectrogram(spectrogram):
    """
    Normalizes the spectrogram values to the range 0-255 for image display.

    Parameters:
        spectrogram (np.ndarray): 2D array of spectrogram amplitude values.

    Returns:
        np.ndarray: Normalized 2D array with type uint8.
    """
    # Shift values so that the minimum is zero
    spec_norm = spectrogram - np.min(spectrogram)
    # Scale to the range 0-255
    spec_norm = spec_norm / np.max(spec_norm) * 255
    return spec_norm.astype(np.uint8)

def save_spectrogram_image(signal, sample_rate, output_file,
                           window_size=1024, hop_size=512):
    """
    Computes the spectrogram of an audio signal and saves it as an image.

    Parameters:
        signal (np.ndarray): 1D array of audio samples.
        sample_rate (int): Sampling rate of the audio signal (unused in image creation).
        output_file (str): Filename for saving the spectrogram image.
        window_size (int): Number of samples per frame.
        hop_size (int): Number of samples to shift between frames.
    """
    # Compute the spectrogram
    spectrogram = compute_spectrogram(signal, window_size, hop_size)

    # Normalize to the range 0-255 for image conversion
    spec_image = normalize_spectrogram(spectrogram)

    # Create an image from the numpy array.
    # The spectrogram's first row corresponds to the lowest frequency.
    # Flip vertically to have low frequencies at the bottom.
    img = Image.fromarray(spec_image)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the image
    img.save(output_file)
    print(f"Spectrogram image saved to '{output_file}'.")

def process_wav_file(filename, output_file, window_size=1024, hop_size=512):
    """
    Reads a WAV file, computes its spectrogram, and saves the spectrogram as an image.

    Parameters:
        filename (str): Path to the WAV file.
        output_file (str): Path where the output image will be saved.
        window_size (int): Number of samples per frame.
        hop_size (int): Number of samples to shift between frames.
    """
    sample_rate, data = wav.read(filename)

    # If stereo (or multi-channel), take only the first channel.
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize integer data to float if needed.
    if data.dtype not in [np.float32, np.float64]:
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val

    print(f"Processing '{filename}' (sample rate: {sample_rate} Hz)...")
    save_spectrogram_image(data, sample_rate, output_file, window_size, hop_size)

if __name__ == "__main__":
    # Process the "record.wav" file and save the spectrogram image as "spectrogram.png"
    process_wav_file("record.wav", "custom_spectrogram.png")
