import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def draw_spectrogram(audio_file_path, image_file_path,
                     min_hz=None, max_hz=None, min_db=None, max_db=None):
    # calculate spectrogram
    sample_rate, audio_data = wavfile.read(audio_file_path)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    sample_frequencies, segment_times, spectrogram_data = spectrogram(audio_data, fs=sample_rate)

    # apply frequency range masks
    spectrogram_data_db = 10 * np.log10(spectrogram_data + 1e-10)
    freq_mask = np.ones_like(sample_frequencies, dtype=bool)
    if min_hz is not None:
        freq_mask &= (sample_frequencies >= min_hz)
    if max_hz is not None:
        freq_mask &= (sample_frequencies <= max_hz)
    sample_frequencies = sample_frequencies[freq_mask]
    spectrogram_data_db = spectrogram_data_db[freq_mask, :]

    # apply power range masks
    vmin = min_db if min_db is not None else np.min(spectrogram_data_db)
    vmax = max_db if max_db is not None else np.max(spectrogram_data_db)

    # draw spectrogram
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram_data_db, aspect='auto', origin='lower',
               extent=[segment_times.min(), segment_times.max(), sample_frequencies.min(), sample_frequencies.max()],
               cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')

    # save image
    plt.savefig(image_file_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    draw_spectrogram('record.wav', 'spectrogram.png', max_hz=2500, min_db=-60)
