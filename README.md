📊 YouTube Comments Sentiment & Abusive Language Analysis

An interactive Machine Learning and NLP-based web application that analyzes YouTube video comments to identify sentiment polarity and abusive/offensive language.
Built using Streamlit, classical ML models, and transformer-based deep learning.

🚀 Project Overview

This project fetches comments from YouTube videos using the YouTube Data API, preprocesses multilingual text, and applies Natural Language Processing (NLP) and Machine Learning techniques to:

Classify comments as Positive, Negative, or Neutral

Detect abusive/offensive comments

Compare the performance of multiple ML models

Visualize sentiment trends using charts and word clouds

The application provides a user-friendly web interface for real-time analysis.

✨ Key Features

📥 Fetch up to 500 YouTube comments per video

🌐 Automatic language detection & translation to English

🧹 Advanced text preprocessing and cleaning

😊 Sentiment classification (Positive / Negative / Neutral)

🚨 Abusive language detection using RoBERTa Transformer

🤖 Multiple ML models with hyperparameter tuning

📊 Model comparison and performance evaluation

☁️ Word cloud visualization

🖥️ Interactive Streamlit UI

🛠️ Tech Stack

Programming & Frameworks

Python

Streamlit

Machine Learning & NLP

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

🧠 Machine Learning Workflow

Collect YouTube comments using API

Clean and preprocess text data

Detect language and translate non-English text

Extract features using TF-IDF (word + character n-grams)

Handle class imbalance using ADASYN

Train and tune ML models using GridSearchCV

Evaluate using accuracy, confusion matrix, and classification report

🧪 Application Modules
1️⃣ Single Video Analysis

Sentiment and abusive content analysis

Train and evaluate selected ML models

2️⃣ Compare Two Videos

Compare sentiment distributions between two videos

Model performance comparison

3️⃣ Compare Two Models

Side-by-side ML model evaluation on the same video

4️⃣ Abusive Comments Detection

Dedicated tab to view offensive comments

Word cloud of abusive comments

🖥️ How to Run the Project
🔧 Prerequisites

Python 3.8+

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

Ensemble and XGBoost models showed strong performance

TF-IDF word + character features improved robustness

Transformer-based abusive detection increased accuracy

Effective handling of multilingual comments

🌱 Future Enhancements

Fine-tuned transformer-based sentiment model

Real-time comment streaming

Topic modeling and clustering

Cloud deployment

👩‍💻 Author

Tapasvini S
🎓 MSc Artificial Intelligence & Machine Learning
📧 Email: your-email@example.com

🔗 GitHub: https://github.com/your-username

⭐ Acknowledgements

Hugging Face Transformers

Google YouTube Data API

NLTK & Scikit-learn Community
