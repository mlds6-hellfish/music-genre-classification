from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, ClassifierMixin

def generate_model_pipeline(
    extractor: ColumnTransformer,
    selector: TransformerMixin,
    classifier: ClassifierMixin
    ) -> Pipeline:
    """
    Function to get sklearn classification pipeline
    Parameters
    ----------
        extractor (ColumnTransformer): [description]
        selector (TransformerMixin): [description]
        classifier (ClassifierMixin): [description]

    Returns
    -------
        pipeline : Pipeline
            sklearn Pipeline obj
    """
    
    pipeline = Pipeline(
        steps=[
            ("extractor", extractor),
            ("selector", selector),
            ("classifier", classifier)
        ]
    )

    return pipeline
