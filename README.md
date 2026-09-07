IMDb Movie Reviews Sentiment Analysis

A Natural Language Processing (NLP) and Machine Learning project that classifies IMDb movie reviews as Positive or Negative.

The notebook covers exploratory data analysis, text preprocessing, multiple feature-extraction techniques, machine-learning models, model comparison, and final sentiment prediction.

Project Overview

The goal of this project is to build and compare text-classification models for IMDb movie reviews and identify the best-performing approach.

The project explores:

Text cleaning and normalization

Tokenization

Stopword removal

Stemming and lemmatization

Count Vectorization

TF-IDF Vectorization

N-gram features

Word2Vec embeddings

Logistic Regression

Multinomial Naive Bayes

Linear SVM

Random Forest

Model evaluation and comparison

Sentiment prediction on new reviews

Dataset

Dataset: IMDb Dataset of 50K Movie Reviews

Source: Kaggle
Dataset: IMDb Dataset of 50K Movie Reviews

Dataset Information

Property

Details

Total reviews

50,000

Features

review, sentiment

Target classes

Positive, Negative

Positive reviews

25,000

Negative reviews

25,000

Unique reviews

49,582

The dataset is balanced, with 25,000 positive and 25,000 negative reviews.

Project Workflow

IMDb Dataset
     ↓
Exploratory Data Analysis
     ↓
Duplicate Removal
     ↓
HTML Tag Removal
     ↓
Text Cleaning
     ↓
Lowercasing
     ↓
Punctuation Removal
     ↓
Tokenization
     ↓
Stopword Removal
     ↓
Stemming / Lemmatization
     ↓
Clean Review Text
     ↓
Train-Test Split (80:20)
     ↓
Feature Extraction
     ├── CountVectorizer
     ├── TF-IDF
     ├── N-grams
     └── Word2Vec
     ↓
Model Training
     ├── Logistic Regression
     ├── Multinomial Naive Bayes
     ├── Linear SVM
     └── Random Forest
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Sentiment Prediction

Exploratory Data Analysis

The dataset was inspected for:

Dataset shape and structure

Data types

Missing values

Duplicate records

Sentiment distribution

Review length

Processed text representations

No missing values were present in the original dataset. Duplicate records were removed during preprocessing.

After duplicate removal, the working dataset contains 49,582 reviews.

Text Preprocessing

The following preprocessing operations are implemented in the notebook:

Remove duplicate records

Remove HTML tags

Convert text to lowercase

Remove punctuation

Tokenize reviews using NLTK

Remove English stopwords

Apply Porter stemming

Apply WordNet lemmatization

Join processed tokens into cleaned review text

The final Clean_Review representation is used as the main input for feature extraction and model training.

Feature Extraction

1. CountVectorizer

A unigram CountVectorizer was used with:

Maximum features: 20,000

N-gram range: (1, 1)

2. TF-IDF

TF-IDF was used with:

Maximum features: 30,000

N-gram range: (1, 2)

sublinear_tf=True

3. N-gram Features

A CountVectorizer using unigrams and bigrams was also evaluated:

Maximum features: 20,000

N-gram range: (1, 2)

4. Word2Vec

A Word2Vec model was trained using the lemmatized reviews.

Configuration used in the notebook:

Vector size: 100

Window: 5

Minimum word count: 2

Workers: 4

Training algorithm: CBOW (sg=0)

The resulting Word2Vec vocabulary contains 74,198 words.

Machine Learning Models

The project evaluates four classical machine-learning algorithms:

Model

Description

Logistic Regression

Linear classifier suitable for high-dimensional text features

Multinomial Naive Bayes

Probabilistic classifier commonly used for text classification

Linear SVM

Margin-based linear classifier effective for sparse text features

Random Forest

Ensemble of decision trees

Model Performance

CountVectorizer Results

Model

Accuracy

Logistic Regression

87.73%

Multinomial Naive Bayes

85.12%

Linear SVM

85.08%

Random Forest

85.44%

TF-IDF Results

Model

Accuracy

Logistic Regression

89.64%

Multinomial Naive Bayes

87.88%

Linear SVM

89.67%

Random Forest

85.60%

N-gram and Word2Vec Results

Approach

Accuracy

N-gram Linear SVM

86.65%

Word2Vec + Logistic Regression

85.80%

Best Model

Based on the experiments in the notebook, the Linear SVM with TF-IDF features achieved the highest accuracy:

Accuracy: 89.67%

Best Model: Linear SVM
Feature Extraction: TF-IDF
Accuracy: 89.67%

The notebook uses this model as the final classifier for sentiment prediction.

Evaluation Metrics

Model performance is evaluated using:

Accuracy

Confusion Matrix

Precision

Recall

F1-score

Classification Report

The final model's predictions are also evaluated using a classification report and confusion matrix.

Train-Test Split

The dataset is divided using an 80:20 train-test split:

Training set: 80%

Testing set: 20%

random_state=42

Stratified split to preserve the class distribution

Sentiment Prediction

The notebook includes a prediction function for classifying new movie reviews.

The prediction pipeline applies the same core preprocessing steps to the input review before transforming it with the trained TF-IDF vectorizer and passing it to the selected Linear SVM model.

Example:

review = "This movie was absolutely fantastic and I really enjoyed it."

prediction = predict_sentiment(review)
print(prediction)

Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

NLTK

Scikit-learn

Gensim

TensorFlow / Keras

Google Colab

Project Structure

IMDb_Movie_Reviews/
│
├── IMDb_Movie_Reviews.ipynb
├── README.md
└── dataset/
    └── IMDB Dataset.csv

How to Run

1. Clone the repository

git clone https://github.com/tojotk/IMDb_Movie_Reviews.git
cd IMDb_Movie_Reviews

2. Install dependencies

pip install pandas numpy matplotlib seaborn nltk scikit-learn gensim tensorflow

3. Download NLTK resources

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

4. Open the notebook

Run:

IMDb_Movie_Reviews.ipynb

using Google Colab or Jupyter Notebook.

Note: The notebook currently loads the dataset from a Google Drive path. If running outside the original Colab environment, update the dataset path accordingly.

Team Members

Name

Contribution

Tojo Tom

Data Preprocessing

Esha Manohar

NLP Model Building

Key Takeaways

The dataset contains an equal number of positive and negative reviews.

Text preprocessing substantially prepares the raw reviews for machine-learning models.

TF-IDF performed better than the tested CountVectorizer, N-gram, and Word2Vec approaches in the reported experiments.

Among the evaluated TF-IDF models, Linear SVM achieved the highest accuracy of 89.67%.

The final pipeline can be used to predict the sentiment of previously unseen movie reviews.

Future Improvements

Possible extensions include:

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Cross-validation for more robust evaluation

Comparing additional word-embedding approaches

Testing transformer-based models such as BERT

Building a web interface for real-time sentiment prediction

Saving the trained model and vectorizer for deployment
