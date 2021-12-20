from pydantic import BaseModel
from typing import List

class RandomKFoldExp(BaseModel):
    """
    Class for describing a experiment

    Parameters
    ----------
    cv: int
        number of partitions for K-fold
    
    scoring : str
        strategy used for scoring
    """
    cv : int
    # n_iter : int
    scoring : str

class SvmHyperparameters(BaseModel):
    """
    Class to pass data about the model pipeline
    hyperparameters

    Parameters
    ----------
    """
    classifier__C: List[float]
    classifier__gamma: List[float]
    classifier__kernel: List[str]

class RandomForestHyperparameters(BaseModel):
    """
    Class to pass data about the model pipeline

    """

    classifier__max_depth: List[int]
    classifier__min_samples_split: List[int]
    classifier__n_estimators: List[int]