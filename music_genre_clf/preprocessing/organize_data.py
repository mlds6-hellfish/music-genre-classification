import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Optional

def split_and_save_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    test_size: Optional[float]=0.3,
    random_state: Optional[int]=42,
    target_col: Optional[str]='label',
    export_filetype: Optional[str]='csv'
    ):
    """
    Function to partition and save
    datasets (according to the processed data path)
    Note: feats and labels are stored independently
    
    Parameters
    ----------
        df : pd.DataFrame
            Dataset containing all the data
        dataset_name: str
            Name for the generated files
        test_size : Optional[float]
            Size for the test split
            Defaults to 0.3.
        random_state: Optional[int]
            Seed for the random state
            Defaults to 42.
        target_col: Optional[str]
            Col indicating the target variable
            Defaults to label
        export_filetype: Optional[str]
            File extension for the generated
            files to be stored
    """
    
    out_path = os.environ["PROCESSED_DATA_PATH"]

    label = df[target_col]
    feats = df.loc[:, df.columns != target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        feats, label, test_size=test_size, random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    trn_export_fn = eval(f"train_df.to_{export_filetype}")
    tst_export_fn = eval(f"test_df.to_{export_filetype}")
    

    trn_export_fn(f"{os.path.join(out_path, dataset_name)}_trn.{export_filetype}", index=False)
    tst_export_fn(f"{os.path.join(out_path, dataset_name)}_tst.{export_filetype}", index=False)
    

