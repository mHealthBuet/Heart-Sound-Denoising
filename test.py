from wave import open as open_wave
from matplotlib.pylab import frombuffer, specgram
from matplotlib.pyplot import subplot, plot, title, show

def show_wave_n_spec(audiofile_dir: str) -> None:
    """
    Import audio file and show its waveform and
    spectogram representation.

    Args:
        audiofile_dir: directory string name of
            the single heartsound audio file

    Returns: None, waveform and spectogram is displayed
    """

    audiofile_dir = Path(audiofile_dir)
    spf = open_wave(str(audiofile_dir), "r")

    sound_info = spf.readframes(-1)
    sound_info = frombuffer(sound_info, int16)

    subplot(211)
    plot(sound_info)
    title(f'Waveform & spectrogram of "{audiofile_dir.stem}"')

    f = spf.getframerate()
    subplot(212)
    specgram(sound_info, Fs=f, scale_by_freq=True, sides="default")

    show()
    spf.close()
