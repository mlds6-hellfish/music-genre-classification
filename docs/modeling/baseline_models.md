# Baseline Model Report

_Baseline model is the the model a data scientist would train and evaluate quickly after he/she has the first (preliminary) feature set ready for the machine learning modeling. Through building the baseline model, the data scientist can have a quick assessment of the feasibility of the machine learning task._

> If using the Automated Modeling and Reporting tool, most of the sections below will be generated automatically from this tool. 

## Analytic Approach

The objective of the model is to classify music audio tracks on 10 different possible genres. To do so, the model is contructed as a <i>Convolutional Neural Network</i>, whose input are images of spectrograms of shape (1025,n), and the output are 10 nodes to represent each genre. 

## Model Description

	
The data is received as audio files in 22050 Hz Mono 16-bit .wav format, and through a pre-processing pipe, namely, the absolute value of short time Fourier transform, an spectrogram is generated. This is the input for the convolutional neural network.

The learner optimizes categorical cross entropy, using ADAM. Regularizers L1 and L2 will be used on every middle layer, and each respective lambda. Dropouts on each middle layer with a set weight ‘p’. 

The hyperparameters of the learner are: ADAM’s learning rate, for all regularizers, each weight lambda,  the dropout rate ‘p’.