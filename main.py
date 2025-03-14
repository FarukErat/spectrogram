import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def save_spectrogram_image(wav_file, output_image):
    # Read WAV file
    sample_rate, data = wav.read(wav_file)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Create the spectrogram
    plt.figure(figsize=(10, 5))
    plt.specgram(data, NFFT=1024, Fs=sample_rate, cmap='inferno', noverlap=512)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')

    # Save the figure instead of displaying it
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
save_spectrogram_image('record.wav', 'spectrogram.png')
