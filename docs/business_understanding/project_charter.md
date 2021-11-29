# Project Charter

## Business background

* This project is designed for music reproduction services (or music genre recognition).
* Manual classification of music genres is a difficult and time consuming task. Being able to automate such a task can provide multiple benefits for organization of songs in music libraries and recommendation systems, in addition to potentially providing a starting point for extracting insights (e.g., trends) from music-related data.

## Scope
* Data science solution: Music genre classifier.
* An application capable of classifying audio clips extracted from songs into music genres.
* Customer access: The product will be available to customers via a web application.


## Personnel
* Who are on this project:
	* Hellfish:
		* Project lead: Juan Lara
		* PM: Melissa de la Pava
		* Data scientist(s): Daniel Palleja, Alvaro Rodriguez, Juan Pablo Zuluaga, Ricardo Alejandro Orjuela.

	* Client:
		* Data administrator: Melissa de la Pava
		* Business contact: Melissa de la Pava
	
## Metrics
* What are the qualitative objectives? Reduce the difficulty in the recognition of musical genres and automatic feature extraction.
* What is a quantifiable metric  cosntruct a model for the recognition of musical genres and implement it in an application 

## Plan
This project will be divided into three predefined phases with stakeholders:
* Phase 1 (Business Understanding): 22 nov - 28 nov. Read documentation on the state of the art, do exploratory data analysis on the datasets.
* Phase 2 (Preprocessing-Modelling):29 nov - 5 dec. Design of preprocessing pipeline and modeling of the classificator.
* Phase 3 (Implementation): 6 dec - 12 dec. Implementation of the model and deployment via a web application


## Architecture
* Data
	* What data do we expect? Ours raw data will be downloaded from the website (http://marsyas.info/downloads/datasets.html) in the link (http://opihi.cs.uvic.ca/sound/genres.tar.gz)
	* This dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050 Hz Mono 16-bit audio files in .wav format

* Data movement from on-prem to Azure using ADF or other data movement tools (Azcopy, EventHub etc.) to move either
  * all the data, 
  * after some pre-aggregation on-prem,
  * Sampled data enough for modeling 

* What tools and data storage/analytics resources will be used in the solution e.g.,
  * Colab Notebooks python for preprocessing, feature construction, aggregation and sampling.
  * Libraries as Numpy, scikitLearn, Librosa, Matplotlib
  * Python as programming language
* How will the score or operationalized web service(s) (RRS and/or BES) be consumed in the business workflow of the customer? If applicable, write down pseudo code for the APIs of the web service calls.
  * The product will be consumed through a web application that will be built throughout the courses

## Communication
* How will we keep in touch? Weekly meetings?
	* We will keep in touch via slack in addition to short meetings three times per week.
* Who are the contact persons on both sides?
	* *Hellfish:* Daniel Palleja, Alvaro Rodriguez, Juan Pablo Zuluaga, Ricardo Alejandro Orjuela
	* *Client:* Melissa de la Pava, Juan Sebastián Lara
