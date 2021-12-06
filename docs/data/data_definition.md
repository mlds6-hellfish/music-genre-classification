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

The Feature sets will be constructed by the convolutional neural network. Therefore, there are not any feature set
