import json, os
import pandas as pd
from typing import Dict
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import GridSearchCV

def load_json_file(lake_path: str, file_name: str) -> Dict:
    """
    Function to load a json file as a Dict.
    Parameters
    ----------
    lake_path : str
        Path to the data lake.
    file_name : str
        Name of the file to upload.
    
    Returns
    -------
    Dict
        Loaded file.
    """
    file_path = os.path.join(lake_path, file_name)
    with open(file_path) as f:
        data = json.load(f)
    return data

def pd_load_df(
    base_path: str,
    file_name: str,
    file_type: str,
    **kwargs
    ) -> pd.DataFrame:
    """
    Function to load data in multiple formats
    using pandas' DataFrames.
    Parameters
    ----------
    base_path : str
        Path for the directory containing the data
        to be loaded.
    file_name : str
        File name (without extension)
    file_type : str
        Extension of the file to be loaded.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    
    file = f"{file_name}.{file_type}"
    file_path = os.path.join(base_path, file)
    load_fn = eval(f"pd.read_{file_type}")
    df = load_fn(file_path, **kwargs)
    return df

def export_model(
        model: Pipeline, base_path: str, file_name: str
        ):
    """
    This function exports a model to a file.
    Parameters
    ----------
    model : Pipeline
        Model to export.
    base_path : str
        Path of the models.
    file_name : str
        File name for the model.
    """
    file_path = os.path.join(base_path, file_name)
    joblib.dump(model, file_path)

def load_model(base_path: str, file_name: str):
    """
    This function loads a model from a file.
    Parameters
    ----------
    base_path : str
        Path of the models.
    file_name : str
        File name for the model.
    Returns
    -------
    model : Pipeline
        Loaded model.
    """
    file_path = os.path.join(base_path, file_name)
    model = joblib.load(file_path)
    return model


def export_cv_results(grid_search: GridSearchCV, base_path: str, file_name: str):
    """
    This function exports a model to a file.
    Parameters
    ----------
    model : Pipeline
        Model to export.
    base_path : str
        Path of the models.
    file_name : str
        File name for the model.
    """
    results = pd.DataFrame(grid_search.cv_results_)
    save_path = os.path.join(base_path, file_name)
    results.to_csv(save_path)