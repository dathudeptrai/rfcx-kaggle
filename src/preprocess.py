import os
import glob
import logging
import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000


def find_files(directory, ext="flac"):
    return sorted(glob.glob(directory + f"/**/*.{ext}", recursive=True))


def fast_read_audio(path, target_sample_rate):
    audio, sr = sf.read(path)
    if len(audio.shape) == 2:
        audio = audio[..., 0]
    return audio.astype(np.float32), sr


def read_mfcc(
    input_filename,
    sample_rate,
):
    audio, sr = fast_read_audio(input_filename, sample_rate)
    mfcc = mfcc_fbank(audio, sr)
    mfcc = np.stack([mfcc, mfcc, mfcc], -1)
    return mfcc


class Audio:
    def __init__(
        self,
        audio_dir: str = None,
        sample_rate: int = SAMPLE_RATE,
        ext="flac",
        tag="train",
    ):
        self.ext = ext
        self.tag = tag
        self.audio_dir = audio_dir
        if audio_dir is not None:
            self.build_cache(os.path.expanduser(audio_dir), sample_rate)

    def build_cache(self, audio_dir, sample_rate):
        logger.info(f"audio_dir: {audio_dir}.")
        logger.info(f"sample_rate: {sample_rate:,} hz.")
        audio_files = find_files(audio_dir, ext=self.ext)
        audio_files_count = len(audio_files)
        assert (
            audio_files_count != 0
        ), f"Could not find any {self.ext} files in {audio_dir}."
        logger.info(f"Found {audio_files_count:,} files in {audio_dir}.")

        from functools import partial

        def iterator_data(items_list):
            for item in items_list:
                yield item

        train_iterator_data = iterator_data(audio_files)

        p = Pool(16)

        partial_fn = partial(self.cache_audio_file, sample_rate=sample_rate)
        train_map = p.imap_unordered(
            partial_fn,
            tqdm(
                train_iterator_data,
                total=len(audio_files),
                desc=f"[Preprocessing {self.tag}]",
            ),
            chunksize=10,
        )

        for _ in train_map:
            pass

    def cache_audio_file(self, input_filename, sample_rate):
        mfcc = read_mfcc(
            input_filename,
            sample_rate,
        )
        np.save(
            input_filename.replace(".flac", ".npy").replace(".//", "./"),
            mfcc,
        )


def mfcc_fbank(
    signal: np.array,
    sample_rate: int,
):  # 1D signal array.
    fmin = 93.75
    fmax = 13687.5
    S = librosa.feature.melspectrogram(
        signal,
        sample_rate,
        n_fft=2048,
        hop_length=int(sample_rate * 0.02),
        fmin=fmin,
        fmax=fmax,
        n_mels=128,
    )
    log_melspectrogram = np.log10(np.maximum(S, 1e-10)).T
    return np.array(log_melspectrogram, dtype=np.float32)


if __name__ == "__main__":
    Audio(audio_dir="../../../data/rcfx/train", tag="train", ext="flac")
    Audio(audio_dir="../../../data/rcfx/test", tag="test", ext="flac")
