from numpy import array
from pathlib import Path
from itertools import chain
from keras.models import load_model
from librosa.display import waveshow, specshow
from scipy.io.wavfile import write as write_audio
from matplotlib.pyplot import subplot, title, suptitle, show
from librosa import load as load_audio, stft, amplitude_to_db


class CleanHeartSounds:
    def __init__(self) -> None:
        pass

    def get_audios(self, audiofiles_dir: str, model_input_shape: int) -> None:
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
            audio, sampling_rate = load_audio(audiofile)

            # Split audio in batches/chunks
            for i in range(0, len(audio), model_input_shape):
                next_split = i + model_input_shape
                chunk = audio[i:next_split]

                # The last chunk may not have the
                # shape the model needs
                if chunk.shape[0] == model_input_shape:
                    chunk = chunk.reshape((model_input_shape, -1))
                    self.audios.append(chunk)

                    # Keep track which chunk corresponds
                    # to which original audio
                    self.names.append(audiofile.stem)

        self.audios = array(self.audios)

        # Transforms it to dictionary like:
        # {0:audio1, 1: audio1, ..., j: audio2, ...}
        self.names = dict(enumerate(self.names))

    def predict(self, model_dir: str) -> array:
        """
        Import the model and predict

        Args: model_dir: string directory name where
            the model is placed

        Returns: an array representing the prediction
        """

        self.model = load_model(model_dir)
        clean = self.model.predict(self.audios)
        return clean

    def group_clean(self) -> None:
        """
        Reverses the self.names dict attribute to have
        which indexes of the prediction batches are linked
        to the same original image

        If self.names = {0:audio1, 1:audio1, 2:audio2}
        then self.grouped = {audio1:[0,1], audio2:[2]}
        """

        self.grouped = {}
        for x, y in self.names.items():
            if y not in self.grouped:
                self.grouped[y] = [x]
            else:
                self.grouped[y].append(x)

    def save_clean(self) -> array:
        """
        Creates a prediction folder, join th prediction
        batches and export them
        """

        self.clean_dir = self.audiofiles_dir.joinpath("clean")
        self.clean_dir.mkdir(exist_ok=True)

        for name, indexes in self.grouped.items():
            to_save = self.clean[indexes]

            # Join the list of lists into one
            to_save = list(chain(*to_save))
            to_save = array(to_save)

            new_name = f"{name}_clean.wav"
            new_dir = self.clean_dir.joinpath(new_name)
            write_audio(new_dir, 22050, to_save)

    def clean_heart_sounds(
        self, audiofiles_dir: str, model_dir: str, model_input_shape=int
    ) -> array:
        """
        The one and only method to import data, predict and export results

        Args:
            audiofiles_dir: arg for self.get_audios(...) method
            model_dir: arg for self.predict(...) method
            model_input_shape: arg for self.get_audios(...) method

        Returns: an array representing the prediction
        """

        self.get_audios(audiofiles_dir, model_input_shape)
        self.clean = self.predict(model_dir)
        self.group_clean()
        self.save_clean()
        return self.clean

    def compare_clean(self, audiofile_dir: str) -> None:
        """
        Import audio file and show its original and cleaned
        waveform and spectogram representation.

        Args:
            audiofile_dir: directory string name of
                the single original heartsound audio file

        Returns: None, waveform and spectogram comparison is displayed
        """

        # Original

        orig_dir = Path(audiofile_dir)
        audio_name = orig_dir.stem

        orig_audio, sampling_rate = load_audio(orig_dir)

        subplot(2, 2, 1)
        waveshow(orig_audio, sr=sampling_rate)
        title("Original")

        subplot(2, 2, 3)
        orig_spec = stft(orig_audio)
        orig_spec = amplitude_to_db(abs(orig_spec))
        specshow(orig_spec, sr=sampling_rate)

        # Clean

        clean_name = f"{audio_name}_clean"
        clean_dir = orig_dir.parent.joinpath("clean", clean_name + ".wav")
        clean_audio, sampling_rate = load_audio(clean_dir)

        subplot(2, 2, 2)
        waveshow(clean_audio, sr=sampling_rate)
        title("Clean")

        subplot(2, 2, 4)
        clean_spec = stft(clean_audio)
        clean_spec = amplitude_to_db(abs(clean_spec))
        specshow(clean_spec, sr=sampling_rate)

        suptitle(f'Waveform & spectrogram of "{audio_name}"')
        show()
