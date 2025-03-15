import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import os

def save_spectrogram(filename, output_file, min_hz=None, max_hz=None,
                    min_db=None, max_db=None, dpi=100, format=None):
    """
    Save a spectrogram of a WAV file to an image file with customizable ranges.

    Parameters:
        filename (str): Path to the input WAV file
        output_file (str): Path for the output image file
        min_hz (float): Minimum frequency to display (default: 0 Hz)
        max_hz (float): Maximum frequency to display (default: Nyquist frequency)
        min_db (float): Minimum amplitude in dB (default: actual minimum in data)
        max_db (float): Maximum amplitude in dB (default: actual maximum in data)
        dpi (int): Image resolution (dots per inch)
        format (str): Image format (inferred from extension if None)
    """
    # Read WAV file
    sample_rate, data = wavfile.read(filename)

    # Convert stereo to mono
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    # Compute spectrogram
    nperseg = 1024  # Window size
    noverlap = nperseg // 2  # Overlap between windows
    f, t, Sxx = spectrogram(data, fs=sample_rate, nperseg=nperseg,
                            noverlap=noverlap, window='hann')

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx)

    # Handle -infinity values from log10(0)
    finite_vals = Sxx_db[np.isfinite(Sxx_db)]
    if finite_vals.size == 0:
        raise ValueError("All amplitude values are zero in the spectrogram")
    Sxx_db = np.clip(Sxx_db, a_min=finite_vals.min(), a_max=None)

    # Set default frequency range
    min_hz = 0 if min_hz is None else min_hz
    max_hz = sample_rate / 2 if max_hz is None else max_hz

    # Set default dB range
    min_db = Sxx_db.min() if min_db is None else min_db
    max_db = Sxx_db.max() if max_db is None else max_db

    # Create plot
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(Sxx_db, aspect='auto', origin='lower',
              extent=[t.min(), t.max(), f.min(), f.max()],
              vmin=min_db, vmax=max_db, cmap='inferno')

    plt.ylim(min_hz, max_hz)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Intensity [dB]')

    plt.title(f"Spectrogram of {os.path.basename(filename)}")
    plt.tight_layout()

    # Save to file instead of displaying
    plt.savefig(output_file, bbox_inches='tight', dpi=dpi, format=format)
    plt.close(fig)

# Example usage:
save_spectrogram("record.wav", "spectrogram.png", max_hz=2500, min_db=-15)
