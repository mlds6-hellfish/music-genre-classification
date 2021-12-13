# Data and Feature Definitions

It will be a unique dataset. The source of this dataset is allocated in the website Marsyas. 
Marsyas (Music Analysis, Retrieval and Synthesis for Audio Signals) is an open source software framework for audio processing with specific emphasis on Music Information Retrieval applications.

## Raw Data Sources

| Dataset Name | Original Location   | Destination Location  | Data Movement Tools / Scripts | Link to Report |
| ---:| ---: | ---: | ---: | -----: |
| GTZAN Genre Collection | This dataset is allocated in web MARSYAS and it will be download by http://opihi.cs.uvic.ca/sound/genres.tar.gz command, that provides the webpage | This dataset will be uploaded and explored in the development phase of the project. It will be uploaded to Google Colab and there it will be handled | | |


* Dataset1 summary. <The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050 Hz Mono 16-bit audio files in .wav format.>

## Processed Data
| Processed Dataset Name | Input Dataset(s)   | Data Processing Tools/Scripts | Link to Report |
| ---:| ---: | ---: | ---: | 
| Processed Dataset 1 | GTZAN Genre Collection | Preprocesamiento_Audio.ipynb | |

* Processed Data1 summary. <The Processed Dataset consists in an audio-to-image transformation in order to create an input for a CNN.>

## Feature Sets

The Feature sets will be constructed by the convolutional neural network. Therefore, there are not any feature set.
  However, as a theoretical exercise, an exploration of the characteristics that have a high correlation with the others is proposed using the kaggle dataset [GTZAN dataset (kaggle)](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)  which is based on the original MARSYAS datset. In kaggle there are a file .csv with all characteristics of preprocessing by correlation that we use for exploratory analysis of the characteristics.
  
| Feature set Name | Input Dataset(s)   | Feature Engineering Tools/Scripts | Link to Report |
| ---:| ---: | ---: | ---: | 
| Feature Dataset 1 | [feature_30sec.csv](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) | EDA.ipynb ||

 The correlation matrix of the variables found in the feature data set is shown. It is important to mention that the characteristics have average and variance measurements and both are shown below.
  
  ![image](https://user-images.githubusercontent.com/95109032/145737229-62cd8f33-bd09-40d4-8b4c-9d8bf619026d.png)
