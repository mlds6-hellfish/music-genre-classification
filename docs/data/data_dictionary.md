# Data Dictionary

For this project we will be using the [GTZAN dataset](http://marsyas.info/downloads/datasets.html) considered to be the MINST equivalent for sound data. The dataset was used in the paper  <i>Musical genre classification of audio signals </i> by G. Tzanetakis and P. Cook.

The following sections provide a detailed description of the data.

## Audio data

The original dataset provides a collection of audio files comprising 100 files for 10 music genres listed next.

<ul>
    <li>Blues</li>
    <li>Classical</li>
    <li>Country</li>
    <li>Disco</li>
    <li>Hiphop</li>
    <li>Jazz</li>
    <li>Metal</li>
    <li>Pop</li>
    <li>Reggate</li>
    <li>Rock</li>
</ul>

Each audio file contains 30 seconds of a song. 

## Additional data

As it has been mentioned, the dataset is widely used among the research community. For that reason, extended datasets are available through popular platforms such as Kaggle.  

Specifically,  [this version](!https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) provides signal-processing data, both images (spectrograms) and numeric data.


## Spectrogram images

Following the same structure described for the audio data, each audio fragment has an equivalent image representing their corresponding mel spectrogram.

| ![image](https://drive.google.com/uc?export=view&id=1ynwRdiAJNwa1gT4CB_9p5yE6UnhR4tlb) |
|:--:| 
| Pop song spectrogram |

| ![image](https://drive.google.com/uc?export=view&id=1hJoUIiRRni812DM78hd140oZQly1NTTZ) |
|:--:| 
| Jazz song spectrogram |


## Numeric features

Following with the available signal-processing-related data, the dataset provides a set of attributes, namely:
    
* Length
* Chroma: Data related to the Constant-Q Transform, which transforms the signal data frequency domain.
* RMS:  root-mean-square
    * Mean
    * Variance
* Spectral centroid: Measure used to characterise a spetrum. This feature is commonly associated with the measure of the brightness of a sound.
    * Mean
    * Variance
* Spectral bandwidth: This featured is derived from the variance spectral centroid.
    * Mean
    * Variance