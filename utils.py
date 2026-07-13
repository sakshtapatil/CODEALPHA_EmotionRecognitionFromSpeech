import librosa
import numpy as np


def extract_features(audio_path):

    audio, sample_rate = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    if mfcc.shape[1] > 300:
        mfcc = mfcc[:, :300]
    else:
        padding = 300 - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, padding)),
            mode="constant"
        )

    mfcc = mfcc.reshape(1, 13, 300, 1)

    return mfcc