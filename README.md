📊 YouTube Comments Sentiment & Abusive Language Analysis

An end-to-end Machine Learning and NLP-based web application that analyzes YouTube video comments to identify sentiment trends and abusive language.
Built using Streamlit, classical ML models, and transformer-based deep learning techniques.

🚀 Project Overview

This project extracts comments from YouTube videos using the YouTube Data API, preprocesses multilingual text, and applies Natural Language Processing (NLP) and Machine Learning techniques to:

Classify comments into Positive, Negative, or Neutral

Detect offensive / abusive language

Compare performance of multiple ML models

Visualize insights using charts and word clouds

Compare sentiment trends across videos and models

The application provides an interactive and user-friendly interface for real-time analysis.

✨ Key Features
🔍 Comment Collection

Fetches up to 500 YouTube comments per video

Supports multilingual comments

Automatic language detection and translation to English

🧹 Text Preprocessing

Emoji removal

URL, mention, hashtag, digit, and punctuation removal

Stopword removal

Lemmatization

Repeated character normalization

😊 Sentiment Analysis

Sentiment classes:

Positive

Negative

Neutral

Uses TextBlob polarity scoring

Visual sentiment distribution

🚨 Abusive Language Detection

Transformer-based model:

cardiffnlp/twitter-roberta-base-offensive

Classifies comments as:

Offensive

Not Offensive

Dedicated tab to view abusive comments separately

🤖 Machine Learning Models

Logistic Regression

Random Forest

Decision Tree

Support Vector Machine (SVM)

K-Nearest Neighbors (Euclidean, Minkowski, Cosine)

XGBoost

Ensemble Voting Classifier

📊 Model Optimization & Evaluation

TF-IDF (Word + Character n-grams)

Chi-Square feature selection

ADASYN oversampling for class imbalance

Hyperparameter tuning using GridSearchCV

Stratified K-Fold Cross Validation

Accuracy, Confusion Matrix & Classification Report

📈 Visual Analytics

Sentiment distribution bar charts

Word clouds for:

All comments

Abusive comments

Model accuracy comparison charts

🛠️ Tech Stack
Programming & Frameworks

Python

Streamlit

NLP & Machine Learning

NLTK

TextBlob

Scikit-learn

Imbalanced-learn

XGBoost

Hugging Face Transformers

PyTorch

Visualization

Matplotlib

Seaborn

WordCloud

APIs

YouTube Data API v3

Google Translator API

🧠 Machine Learning Pipeline

Data Collection

Fetch comments using YouTube API

Preprocessing

Cleaning, translation, lemmatization

Feature Engineering

TF-IDF Word n-grams (1–2)

TF-IDF Character n-grams (3–5)

Feature union

Feature Selection

Chi-Square Test

Class Imbalance Handling

ADASYN oversampling

Model Training & Evaluation

GridSearchCV

Stratified K-Fold Cross Validation

Performance metrics

🧪 Application Modules
1️⃣ Single Video Analysis

Sentiment & abusive comment detection

Train a selected ML model or all models

2️⃣ Compare Two Videos

Compare sentiment distribution between two videos

Model performance comparison

3️⃣ Compare Two Models

Side-by-side ML model comparison on the same video

4️⃣ Abusive Comments Detection

Separate tab for offensive comments

Abusive comments word cloud

🖥️ How to Run the Project
🔧 Prerequisites

Python 3.8 or higher

YouTube Data API Key

📦 Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py

📂 Project Structure
youtube-comments-analysis
├── app.py
├── requirements.txt
├── README.md

📌 Results & Insights

Ensemble and XGBoost models achieved strong performance

Character + word TF-IDF improved robustness

Transformer-based abusive detection improved accuracy

Effective handling of multilingual comments

🌱 Future Enhancements

Fine-tuned transformer-based sentiment classifier

Real-time comment streaming

Topic modeling and clustering

User authentication

Cloud deployment

👩‍💻 Author

Tapasvini S
🎓 MSc Artificial Intelligence & Machine Learning

Areas of Interest
Machine Learning, NLP, Prompt Engineering, UI/UX Design (Figma)

⭐ Acknowledgements

Hugging Face Transformers

Google YouTube Data API

NLTK & Scikit-learn communities
