import pandas as pd
import sklearn.model_selection as skms

from typing import Tuple, Optional

from music_genre_clf.utils.io import pd_load_df
from music_genre_clf.data_models.environment import DataLakePaths


def load_feat_signal_dataset(
    data_paths: DataLakePaths,
    track_sec_length: str,
    is_train_split: bool
    ) -> Tuple:
    """
    Load feat_signal dataset according to provided parameters
    Parameters
    ----------
    data_paths : DataLakePaths
        Object containing paths
    track_sec_length : str
        Indication of the dataset to be loaded 
        (3 or 30 secs data)
    is_train_split : bool
        Indicator of which dataset to load
        (train or test)

    Returns
    -------
    Tuple
        (feats, labels)
    """

    split = "trn" if is_train_split else "tst"
    data = pd_load_df(data_paths.processed_data, f"signal_feats_{track_sec_length}s_{split}", "csv")
    X = data.loc[:, data.columns != 'label']
    y = data['label']
    return X, y

    


