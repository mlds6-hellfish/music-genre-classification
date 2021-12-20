from pydantic import BaseModel

class DataLakePaths(BaseModel):
    """
    Class for accessing data paths.
    Attributes
    ----------
    raw_data: str
        Directory to store raw data.
    processed_data: str
        Directory to store processed data.
    complete_data: str
        Directory to store processed data.
    audio_data: str
        Directory to store audio data.
    img_data: str
        Directory to store image data.
    models: str
        Directory to store models' data.
    results: str
        Folter to save extracted features.

    """
    raw_data: str
    processed_data: str
    complete_data: str
    audio_data: str
    img_data: str
    models: str
    results: str