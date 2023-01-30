from pathlib import Path
from numpy import array, int16
from wave import open as open_wave
from keras.models import load_model
from librosa import load as load_audio
from matplotlib.pylab import frombuffer, specgram
from matplotlib.pyplot import subplot, plot, title, show


def get_audiofiles(audiofiles_dir: str) -> array:
    """
    Import audiofiles and the whole array of arrays

    Args:
        audiofiles_dir: directory string name containing
            all .wav heartsound audio files

    Returns: An array of audio arrays
    """

    audiofiles_dir = Path(audiofiles_dir)

    audios = []
    for audiofile in audiofiles_dir.glob("*.wav"):
        audio = load_audio(audiofile)
        audios.append(audio[0])

    return array(audios)


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


def clean_heartsounds(audiofiles_dir: str) -> array:
    """
    The one and only function you need to import in order
    to clean a directory full of .wav heartsound audio files

    Args:
        audiofiles_dir: directory string name containing
            all .wav heartsound audio files

    Returns:
        An array representing every clean heartsound
    """

    X = get_audiofiles(audiofiles_dir)
    model = load_model("Models/LU-Net.h5")

    clean = model.predict(X)

    return clean


y = clean_heartsounds(audiofiles_dir="Data/predict")

X = get_audiofiles(audiofiles_dir="Data/predict")
model = load_model("Models/LU-Net.h5")
clean = model.predict(X)
