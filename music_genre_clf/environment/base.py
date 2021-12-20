import os, json
from music_genre_clf.data_models.environment import DataLakePaths
from music_genre_clf.data_models.experiments import \
    RandomKFoldExp, SvmHyperparameters, RandomForestHyperparameters


def get_data_paths() -> DataLakePaths:
    """
    Function to get the paths containing data
    Returns
    -------
        paths: DataLakePaths
            Object containing data-reled paths
    """
    paths = DataLakePaths(
        raw_data = os.environ["RAW_DATASET_PATH"],
        processed_data = os.environ["PROCESSED_DATA_PATH"],
        complete_data = os.environ["COMPLETE_DATA_PATH"],
        audio_data = os.environ["AUDIO_DATA_PATH"],
        img_data = os.environ["IMG_DATA_PATH"],
        models = os.environ["MODELS_PATH"],
        results = os.environ["RESULTS_PATH"]
    )

    return paths

def get_k_fold_params() -> RandomKFoldExp:
    path = os.path.join(
            os.environ["EXPERIMENTS_PATH"],
            "random_kfold.json"
            )
    with open(path) as f:
        random_kfold = json.load(f)
    experiment = RandomKFoldExp(
            **random_kfold
            )
    return experiment

def get_svm_hyperparams() -> SvmHyperparameters:
    """[summary]

    Returns
    -------
    SvmHyperparameters
        [description]
    """
    path = os.path.join(
            os.environ["EXPERIMENTS_PATH"],
            "svm_params.json"
            )
    with open(path) as f:
        hyperparameters = json.load(f)
    hyperparameters = SvmHyperparameters(
            **hyperparameters
            )
    return hyperparameters

def get_rand_forest_hyperparams() -> RandomForestHyperparameters:
    path = os.path.join(
            os.environ["EXPERIMENTS_PATH"],
            "randforest_params.json"
            )
    with open(path) as f:
        hyperparameters = json.load(f)
    hyperparameters = RandomForestHyperparameters(
            **hyperparameters
            )
    return hyperparameters