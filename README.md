# Machine Learning Internship Projects at Cantilever

This repository contains the projects completed during my one-month Machine Learning Internship at Cantilever (October–November 2025).  
The internship provided hands-on exposure to real-world machine learning workflows and practical problem-solving in a professional environment.

---

## Projects Overview

| Project No. | Title | Description |
|-------------|--------|-------------|
| 1 | Sentiment Analysis System | NLP-based classifier that detects positive or negative sentiment in movie reviews |
| 2 | [Coming Soon] | Details will be added soon |

---

## Project 1: Sentiment Analysis System

This project implements a complete workflow for sentiment classification using Natural Language Processing techniques. The goal is to train a machine learning model that predicts whether a given text review is positive or negative.

### Technologies Used

| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| NLP Toolkit | NLTK |
| Data Processing | Pandas, NumPy |
| Development Environment | JupyterLab |

---

### Process Overview

**1. Data Preprocessing**  
- Used the NLTK Movie Reviews dataset containing 2000 labeled reviews  
- Performed tokenization, stopword removal and lemmatization  

**2. Feature Extraction**  
- Converted text into TF-IDF numerical vectors  

**3. Model Training**  
- Trained a LinearSVC (Support Vector Machine) model on 80% of the dataset  

**4. Evaluation**  
- Achieved an accuracy of **87.50%** on the remaining 20% test data  

---

## How to Run the Project

```bash
# Install required libraries
pip install jupyterlab scikit-learn nltk pandas numpy

# Clone the repository
git clone <your-repo-link>

# Navigate to the project folder
cd Sentiment-Analysis

# Launch JupyterLab
py -m jupyterlab

