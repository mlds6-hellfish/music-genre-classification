from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from music_genre_clf.data_models.experiments import SvmHyperparameters, RandomKFoldExp

def generate_grid_search(
    pipe: Pipeline,
    hyperparameters: SvmHyperparameters,
    random_kfold_experiment: RandomKFoldExp
):
    """
    Build the grid search object
    Parameters
    ----------
    pipe : Pipeline
        [description]
    hyperparameters : SvmHyperparameters
        [description]
    random_kfold_experiment : RandomKFoldExp
        [description]
    """
    grid_search = GridSearchCV(
        estimator=pipe, param_grid=hyperparameters.dict(),
        refit=True, verbose=3,
        **random_kfold_experiment.dict()
    )

    return grid_search



