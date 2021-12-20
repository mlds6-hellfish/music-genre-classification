import pandas as pd
from typing import List
    

def drop_unnecessary_cols(
    df : pd.DataFrame,
    cols_to_remove: List)->pd.DataFrame:
    """
    Drop the columns that do not provide any interesting 
    info for the models

    Parameters
    ----------
    df: DataFrame
        asdfasdf


    Returns:

    pd.DataFrame: [description]
    """
    
    return df.drop(cols_to_remove, axis=1)