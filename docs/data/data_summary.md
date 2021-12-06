# Data Report

This document contains the results from the exploratory data analysis.

## General summary of the data

The dataset provides images and audio for 10 music genres, comprising 100 files for each genre. The original dataset provides only audio data, however the kaggle version, provides <i>mel spectrograms</i> for each one of the available audio fragments.


## Data quality summary

As the data is publicly available through a [kaggle dataset](!https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) the data quality is good.

It is important to remark that even though we have available data for training the classifier model (in terms of spectrogram images), we are also developing a software module in order to process audio files and extract the spectrogram. This module  is going to be used in the web application.

## Target variable

In this case we are approaching a classification problem with multiple classes. The target variable corresponds to the music genre of the audio fragment at hand (10 classes in total). 

 

## Individual variables

In our case the data does not consist of individual variables but images instead.

| ![image](https://drive.google.com/uc?export=view&id=1ynwRdiAJNwa1gT4CB_9p5yE6UnhR4tlb) |
|:--:| 
| Pop song spectrogram |

