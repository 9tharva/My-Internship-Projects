# Machine Learning Internship Projects at Cantilever

This repository contains the projects completed during my one-month Machine Learning Internship at Cantilever (October–November 2025).
The internship provided hands-on exposure to real-world machine learning workflows and practical problem-solving in a professional environment.

## 📂 Projects Overview

| Project No. | Title | Description |
| :--- | :--- | :--- |
| 1 | **Sentiment Analysis System** | NLP-based classifier that detects positive or negative sentiment in movie reviews. |
| 2 | **Credit Card Fraud Detection** | A classification model to detect fraudulent credit card transactions from a highly imbalanced dataset. |

---

## Project 1: Sentiment Analysis System

This project implements a complete workflow for sentiment classification using Natural Language Processing techniques. The goal is to train a machine learning model that predicts whether a given text review is positive or negative.

### Technologies Used

| Category | Tools |
| :--- | :--- |
| **Programming Language** | Python |
| **Machine Learning** | Scikit-learn (LinearSVC, TfidfVectorizer) |
| **NLP Toolkit** | NLTK |
| **Data Processing** | Pandas, NumPy |
| **Development Environment** | JupyterLab |

### Process Overview

1.  **Data Preprocessing:** Used the NLTK Movie Reviews dataset containing 2000 labeled reviews. Performed tokenization, stopword removal, and lemmatization.
2.  **Feature Extraction:** Converted text into TF-IDF (Term Frequency-Inverse Document Frequency) numerical vectors.
3.  **Model Training:** Trained a **LinearSVC (Support Vector Machine)** model on 80% of the dataset.
4.  **Evaluation:** Achieved an accuracy of **87.50%** on the remaining 20% test data.

---

## Project 2: Credit Card Fraud Detection

This project implements a model to identify fraudulent credit card transactions. A key challenge was handling the **highly imbalanced dataset**, where fraudulent transactions (Class 1) are a very small minority compared to legitimate ones (Class 0).

### Technologies Used

| Category | Tools |
| :--- | :--- |
| **Programming Language** | Python |
| **Machine Learning** | Scikit-learn (Logistic Regression, train_test_split) |
| **Data Processing** | Pandas, NumPy |
| **Data Source** | Kaggle API |
| **Development Environment** | JupyterLab |

### Process Overview

1.  **Data Loading:** Used the Kaggle API to download the dataset.
2.  **Handling Imbalance:** The dataset was found to be highly unbalanced. To address this, an **under-sampling** technique was applied. A new, balanced dataset was created by taking all 492 fraudulent transactions and randomly sampling 492 legitimate transactions.
3.  **Data Splitting:** The new, balanced dataset (of 984 samples) was split into features (X) and targets (Y), and then further split into training (80%) and testing (20%) sets.
4.  **Model Training:** Trained a **Logistic Regression** model on the balanced training data.
5.  **Evaluation:** The model was evaluated for accuracy on the test data to check its performance in classifying unseen transactions.

---

## 🚀 How to Run the Projects

1.  **Install required libraries:**
    ```bash
    # This command installs all necessary packages for both projects
    pip install jupyterlab scikit-learn nltk pandas numpy kaggle
    ```

2.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    ```

3.  **Navigate to the project folder:**
    ```bash
    cd <your-repo-name>
    ```

4.  **Launch JupyterLab:**
    ```bash
    py -m jupyterlab
    ```

### Project-Specific Setup

* **For Sentiment Analysis:** When you first run the `SentimentAnalysis.ipynb` notebook, you will need to run the cells containing `nltk.download(...)` to download the `movie_reviews`, `stopwords`, and `wordnet` datasets.

* **For Credit Card Fraud Detection:** This project requires the Kaggle API.
    1.  Go to your Kaggle account, navigate to "Settings," and click "Create New API Token."
    2.  This will download a `kaggle.json` file.
    3.  Place this `kaggle.json` file in the same directory as the `Credit_Card_Fraud_Detection.ipynb` notebook before running it.
=======
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
