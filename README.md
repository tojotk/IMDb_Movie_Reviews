# IMDb Movie Reviews Sentiment Analysis

## Project Overview

This project performs **Sentiment Analysis** on the IMDb Movie Reviews dataset using Natural Language Processing (NLP) and Machine Learning techniques. The objective is to classify movie reviews as **Positive** or **Negative** based on their textual content.

The project includes data preprocessing, feature extraction using TF-IDF, training multiple machine learning models, and evaluating their performance to identify the most accurate classifier.

---

## Dataset

**Dataset:** IMDb Movie Reviews Dataset

- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

### Dataset Information

- **Total Reviews:** 50,000
- **Classes:** Positive and Negative
- **Features:**
  - Review (Text)
  - Sentiment (Target)

---

# Preprocessing Pipeline

The following preprocessing steps were performed before model training:

1. Load the dataset
2. Remove duplicate records
3. Handle missing values
4. Remove HTML tags
5. Convert text to lowercase
6. Remove punctuation
7. Tokenize the text
8. Remove stopwords
9. Apply stemming
10. Apply lemmatization
11. Join cleaned tokens back into sentences
12. Split the dataset into training and testing sets
13. Convert text into numerical vectors using **TF-IDF Vectorization**

---

# Models Used

The following Machine Learning algorithms were implemented:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Random Forest Classifier

The models were trained using TF-IDF features and evaluated using classification metrics.

---

# Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

---

# How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/IMDb_Movie_Reviews.git
```

### 2. Navigate to the project directory

```bash
cd IMDb_Movie_Reviews
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

### 4. Download NLTK resources

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

### 5. Open the notebook

Run:

```
IMDb_Movie_Reviews.ipynb
```

using Google Colab or Jupyter Notebook.

---

# Project Structure

```
IMDb_Movie_Reviews/
│
├── IMDb_Movie_Reviews.ipynb
├── README.md
└── dataset/
    └── IMDB Dataset.csv
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn

---

# Team Members (Optional)

| Name | Contribution |
|------|--------------|
| Tojo Tom | Data Preprocessing |
| Esha Manohar | NLP Model Building |


---

# Project Workflow

```
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Sentiment Prediction
```

---

# License

This project is intended for educational and academic purposes.
