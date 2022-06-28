import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
import torchaudio.transforms as AT

samplerate = 24000


def age_process(df) -> pd.DataFrame:
    df.loc[df.age <= 10, 'age'] = 0
    df.loc[(df.age > 10) & (df.age <= 20), 'age'] = 1
    df.loc[(df.age > 20) & (df.age <= 30), 'age'] = 2
    df.loc[(df.age > 30) & (df.age <= 40), 'age'] = 3
    df.loc[(df.age > 40) & (df.age <= 50), 'age'] = 4
    df.loc[(df.age > 50) & (df.age <= 60), 'age'] = 5
    df.loc[(df.age > 60) & (df.age <= 70), 'age'] = 6
    df.loc[(df.age > 70), 'age'] = 7
    return df


def gender_process(df) -> pd.DataFrame:
    df.loc[df.gender == 'male', 'gender'] = 1
    df.loc[df.gender == 'female', 'gender'] = 2
    df.loc[df.gender == 'other', 'gender'] = 3
    return df


def get_waveform(audio):
    waveform, _ = torchaudio.load(audio)
    return waveform


def get_melspectogram(audio):
    n_fft = 400
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = AT.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        # pad=10,
        pad_mode="reflect",
        power=2.0,
        onesided=True,
    )
    return mel_spectrogram(get_waveform(audio))


def mfcc(audio):
    return librosa.feature.mfcc(np.array(audio), sr=samplerate, n_mels=128)


def zero_pad(audio, length_signal=5):
    audio = torch.Tensor(audio)
    target = len(audio[0][0][0])

    if length_signal > target:
        missing_sample = length_signal - target
        last_dim_padding = (0, missing_sample)
        return torch.nn.functional.pad(audio, last_dim_padding, value=0)

    else:
        return torch.Tensor(audio[:, :, :5])
