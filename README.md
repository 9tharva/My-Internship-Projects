#Machine Learning Internship Projects @ Cantilever

This repository contains the projects I completed during my one-month Machine Learning Internship at Cantilever from October to November 2025. This opportunity allowed me to apply my skills in a professional environment and contribute to meaningful projects.

##Projects Overview
This repository is organized into folders, with each folder containing a distinct project.

###Sentiment Analysis System: A classical NLP project to classify movie reviews as either positive or negative.
###[Project 2 Title - Coming Soon]: A brief description of your second project will go here.

##1. Sentiment Analysis System
This project is a complete end-to-end pipeline for sentiment analysis, a core task in Natural Language Processing (NLP). The goal is to train a model that can accurately determine the sentiment of a given text.

##Technologies Used
Languages: Python
Libraries: Scikit-learn: For the SVM model and TF-IDF feature extraction.
NLTK (Natural Language Toolkit): For data loading and text preprocessing (tokenization, stop-word removal, lemmatization).
Pandas & NumPy: For data manipulation.
JupyterLab: For interactive development.

##Key Steps & Results
Data Preprocessing: The NLTK movie reviews dataset (2000 reviews) was loaded, cleaned, and lemmatized.
Feature Extraction: Cleaned text was converted into numerical vectors using the TF-IDF technique.
Model Training: A LinearSVC (Support Vector Machine) model was trained on 80% of the data.
Evaluation: The model was tested on the remaining 20% of unseen data, achieving a final accuracy of 87.50%.

##How to Run
Ensure you have Python and the required libraries installed (pip install jupyterlab scikit-learn nltk pandas numpy).

Clone this repository to your local machine.

Navigate to the Sentiment-Analysis folder.

Launch JupyterLab by running py -m jupyterlab in your terminal.

Open the SentimentAnalysis.ipynb notebook and run the cells in order.
