import os
from music_genre_clf.preprocessing.extract_signal_feats import extract_signal_feats
from music_genre_clf.utils.io import load_model

def predict(file_name):
    upload_path = os.environ["UPLOAD_PATH"]
    model_path = os.environ["MODELS_PATH"]
    file = os.path.join(upload_path, file_name)
    audio_fts = extract_signal_feats(file)

    model = load_model(model_path, "signal_feats_3.joblib")
    prediction = model.predict(audio_fts)

    return prediction