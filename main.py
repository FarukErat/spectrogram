import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt

def bandpass_filter(data, sample_rate, min_hz=0, max_hz=10000, order=5):
    nyquist = 0.5 * sample_rate
    low = min_hz / nyquist
    high = max_hz / nyquist

    if min_hz == 0:
        b, a = butter(order, high, btype='low')
    elif max_hz >= nyquist:
        b, a = butter(order, low, btype='high')
    else:
        b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, data)

def save_spectrogram_image(wav_file, output_image, min_hz=0, max_hz=10000, min_db=-80, max_db=0):
    sample_rate, data = wav.read(wav_file)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    filtered_data = bandpass_filter(data, sample_rate, min_hz, max_hz)

    plt.figure(figsize=(10, 5))
    _, _, _, im = plt.specgram(
        filtered_data,
        NFFT=1024,
        Fs=sample_rate,
        cmap='inferno',
        noverlap=512,
        vmin=min_db,
        vmax=max_db
    )

    plt.ylim(min_hz, max_hz)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Spectrogram ({min_hz}-{max_hz} Hz)')
    plt.colorbar(im, label='Intensity (dB)')

    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()

save_spectrogram_image('record.wav', 'spectrogram.png', min_hz=0, max_hz=2000, min_db=0, max_db=50)
