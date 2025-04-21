import plot_spectrogram

plot_spectrogram.plot_spectrogram(
    audio_file_path="record.wav",
    image_file_path="spectrogram.png")

plot_spectrogram.plot_spectrogram(
    audio_file_path="record.wav",
    image_file_path="spectrogram_inverted.png",
    invert=True)

plot_spectrogram.plot_spectrogram(
    audio_file_path="record-2.wav",
    image_file_path="spectrogram_inverted-2.png",
    invert=True)

plot_spectrogram.plot_spectrogram(
    audio_file_path="record.wav",
    image_file_path="spectrogram_high_resolution.png",
    nperseg=2048,
    noverlap=1536)
