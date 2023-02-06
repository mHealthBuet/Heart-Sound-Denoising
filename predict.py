from pathlib import Path
from numpy import array, int16
from wave import open as open_wave
from keras.models import load_model
from librosa import load as load_audio
from scipy.io.wavfile import write as write_audio
from matplotlib.pylab import frombuffer, specgram
from matplotlib.pyplot import subplot, plot, title, show


class CleanHeartSounds:
    def __init__(self) -> None:
        pass


    def get_audiofiles(self, audiofiles_dir: str, model_input_shape: int) -> None:
        """
        Import audiofiles, split them in batches and build the entire 
        array of audio-arrays, keeping a list of its original audiofile
        name and the nth-batch

        Args:
            audiofiles_dir: directory string name containing
                all .wav heartsound audio files
            model_input_shape: shape of the first layer from the
                model that will clean heart sound files

        """

        self.audiofiles_dir = Path(audiofiles_dir)

        self.audios = []
        self.names = []

        for audiofile in self.audiofiles_dir.glob("*.wav"):
            # sampling_rate = 22050
            audio, sampling_rate = load_audio(audiofile)

            for i in range(0, len(audio), model_input_shape):
                next_split = i + model_input_shape
                chunk = audio[i:next_split]

                if chunk.shape[0] == model_input_shape:
                    chunk = chunk.reshape((model_input_shape, -1))
                    self.audios.append(chunk)

                    n_chunk = str(i // model_input_shape + 1)
                    new_name = f'{audiofile.stem}_{n_chunk.zfill(3)}.wav'
                    self.names.append(new_name)

        self.audios = array(self.audios)

    def predict(self, model_dir: str) -> array:
        self.model = load_model(model_dir)
        clean = self.model.predict(self.audios)
        return clean

    def save_clean(self) -> array:
        self.clean_dir = self.audiofiles_dir.joinpath('clean')
        self.clean_dir.mkdir(exist_ok=True)

        for chunk, name in zip(self.clean, self.names):
            write_audio(name, 22050, chunk)

    def clean_heart_sounds(self, audiofiles_dir: str, model_dir: str, model_input_shape= int) -> array:
        self.get_audiofiles(audiofiles_dir, model_input_shape)
        self.clean = self.predict(model_dir)
        self.save_clean()
        return self.clean


chs = CleanHeartSounds()

clean = chs.clean_heart_sounds(
    audiofiles_dir="Data/predict",
    model_dir="Models/LU-Net.h5",
    model_input_shape=800,
)


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

# show_wave_n_spec("Data/predict/b0061.wav")