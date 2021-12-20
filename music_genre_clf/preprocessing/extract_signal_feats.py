import librosa
import numpy as np
import pandas as pd
from typing import Dict


def extract_signal_feats(audio):
    
    # Load audio with librosa
    y, sr = librosa.load(audio)

    # Features
    data = {}
    # Chroma features
    chroma_fts = librosa.feature.chroma_stft(y=y, sr=sr)
    data['chroma_stft_mean'] = [chroma_fts.mean()]
    data['chroma_stft_var'] = [chroma_fts.var()]

    # RMS features
    rms_fts = librosa.feature.rms(y=y)
    data['rms_mean'] = [rms_fts.mean()]
    data['rms_var'] = [rms_fts.var()]

    # Spectral centroid features
    spect_cent_fts = librosa.feature.spectral_centroid(y=y, sr=sr)
    data['spectral_centroid_mean'] = [spect_cent_fts.mean()]
    data['spectral_centroid_var'] = [spect_cent_fts.var()]

    # Spectral bandwidth features
    spec_bw_fts = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    data['spectral_bandwidth_mean'] = [spec_bw_fts.mean()]
    data['spectral_bandwidth_var'] = [spec_bw_fts.var()]

    # Rolloff features
    roll_off_fts = librosa.feature.spectral_rolloff(y=y, sr=sr)
    data['rolloff_mean'] = [roll_off_fts.var()]
    data['rolloff_var'] = [roll_off_fts.var()]

    # Zero crossing rate features
    zero_cross_rt_fts = librosa.feature.zero_crossing_rate(y)
    data['zero_crossing_rate_mean'] = [zero_cross_rt_fts.mean()]
    data['zero_crossing_rate_var'] = [zero_cross_rt_fts.var()]

    # Harmony & Perceptr features
    harmony_fts, perceptr_fts = librosa.effects.hpss(y)
    data['harmony_mean']  = [harmony_fts.mean()]
    data['harmony_var']  = [harmony_fts.var()]

    data['perceptr_mean'] = [perceptr_fts.mean()]
    data['perceptr_var'] = [perceptr_fts.var()]

    # Tempo features
    tempo, _ = librosa.beat.beat_track(y, sr = sr)
    data['tempo'] = [tempo]

    # Length
    data['length'] = [len(y)]

    # Mfcc features
    _get_mfcc_feats(y, sr, data)

    return pd.DataFrame.from_dict(data)


def _get_mfcc_feats(y: np.ndarray, sr: int, data_dict: Dict):

    for i in range(1,21):
        feat_name = f'mfcc{i}'
        mean_name = f'{feat_name}_mean'
        var_name = f'{feat_name}_var'
        
        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=i)

        data_dict[mean_name] = [mfcc.mean()]
        data_dict[var_name] = [mfcc.var()]
        

