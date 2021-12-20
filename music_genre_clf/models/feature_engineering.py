import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2


def get_num_signal_feat_extractor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Function to get the ColumnTransformer obj
    responsible for applying standarization on
    the data

    Parameters
    ----------

    Returns
        transformer: ColumnTransformer
            Object containing the column transformation
            operations

    """

    num_feats = list(df.select_dtypes('number').columns)

    transformer = ColumnTransformer(
        [
            ("num_scale", MinMaxScaler(), num_feats)
        ]
    )

    return transformer


def get_num_feat_selector() -> SelectKBest:
    """
    Function to get the feat selector (k-best)

    Returns
    -------
        SelectKBest: sklearn obj to select feats.
    """
    selector = SelectKBest(chi2)
    return selector