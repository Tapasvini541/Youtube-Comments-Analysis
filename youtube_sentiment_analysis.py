import pandas as pd
import numpy as np
import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE,ADASYN
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from googleapiclient.discovery import build
import streamlit as st
import joblib
from transformers import pipeline, AutoTokenizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import ComplementNB
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from collections import Counter
from sklearn.utils import shuffle
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion
from transformers import BertTokenizer, BertForSequenceClassification,AutoModelForSequenceClassification
import torch

# Load the correct tokenizer
model_name = "cardiffnlp/twitter-roberta-base-offensive"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the model with the correct tokenizer
abusive_detector = pipeline("text-classification", model=model_name, tokenizer=tokenizer,framework="pt")

# Setting seed to ensure reproducibility
DetectorFactory.seed = 0
nltk.download('stopwords')
nltk.download('wordnet')

# Predefined constants
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="YouTube Sentiment Analysis", layout="wide")

# Adding some custom CSS for design
st.markdown(""" 
    <style> 
    .main { background-color: #f0f2f6; } 
    .stTextInput, .stSelectbox, .stButton { margin-bottom: 10px; } 
    .stMetric { font-size: 18px; font-weight: bold; } 
    </style> 
    """, unsafe_allow_html=True)

def demojize_text(text):
    return emoji.demojize(text)

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace="")  # Replace emojis with empty
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\s*\*\s*', ' ', text)  # Remove unnecessary *
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # Limit repeated characters
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Function to Detect and Translate Non-English Text
def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text
        return GoogleTranslator(source=lang, target="en").translate(text)
    except:
        return text

# Function to Get Sentiment (Positive, Negative, Neutral)
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return 'Positive' if polarity >= 0.15 else 'Negative' if polarity <= -0.15 else 'Neutral'

def is_abusive(text):
    encoded_input = tokenizer.encode(text, truncation=True, max_length=512, return_tensors="pt")
    result = abusive_detector(tokenizer.decode(encoded_input[0], skip_special_tokens=True))
    
    if isinstance(result, list) and len(result) > 0:
        return result[0]["label"]  # Return only the label ('OFFENSIVE' or 'NOT_OFFENSIVE')
    
    return "UNKNOWN"  # Fallback if detection fails

# Fetch YouTube Comments Function
@st.cache_data
def fetch_youtube_comments(video_id, api_key):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    video_details = {}
    next_page_token = None

    while len(comments) < 500:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, textFormat="plainText", maxResults=100, pageToken=next_page_token
        )
        response = request.execute()
        comments += [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response.get("items", [])]
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    
    # Fetching Additional Video Statistics (Likes, Views)
    video_request = youtube.videos().list(part="snippet,statistics", id=video_id)
    video_response = video_request.execute()
    
    if video_response.get("items"):
        video_info = video_response["items"][0]
        video_details["likes"] = video_info["statistics"].get("likeCount", 0)
        video_details["views"] = video_info["statistics"].get("viewCount", 0)
        video_details["title"] = video_info["snippet"].get("title", "Unknown")
    
    return comments[:500], video_details

# Preparing the Data for Sentiment Analysis
def prepare_data(comments):
    df = pd.DataFrame(comments, columns=['comment'])
    df = df.dropna().reset_index(drop=True)
    df = df[df['comment'].str.strip() != '']
    df['translated_comment'] = df['comment'].apply(detect_and_translate)
    df['cleaned_comment'] = df['translated_comment'].apply(clean_text)
    df["cleaned_comment"] = df["cleaned_comment"].fillna("").astype(str)  # Convert to string
    df["cleaned_comment"] = df["cleaned_comment"].apply(lambda x: x if len(x.strip()) > 0 else "empty_text")
    df['sentiment'] = df['cleaned_comment'].apply(get_sentiment)
    df["abusive"] = df["cleaned_comment"].apply(is_abusive)
    return df

# Define ML Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000),
    "SVM": SVC(C=1.0, kernel='linear', probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=150, learning_rate=0.1),
    "KNN (Minkowski)": KNeighborsClassifier(n_neighbors=5,metric='minkowski'),
    "KNN (Euclidean)": KNeighborsClassifier(n_neighbors=5,metric='euclidean'),
    "KNN (Cosine)": KNeighborsClassifier(n_neighbors=5,metric='cosine'),
    "Ensemble": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=150)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ], voting='soft')
}

param_grid = {
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "Decision Tree": {"max_depth": [None, 10, 20], "criterion": ["gini", "entropy"]},
    "Logistic Regression": {"C": [0.1, 1, 10], "solver": ['liblinear']},
    "SVM": {"C": [0.1, 1, 10], "kernel": ['linear', 'rbf']},
    "XGBoost": {"max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
    "KNN (Minkowski)": {"n_neighbors": [3, 5], "weights": ['uniform', 'distance']},
    "KNN (Euclidean)": {"n_neighbors": [3, 5], "weights": ['uniform', 'distance']},
    "KNN (Cosine)": {"n_neighbors": [3, 5], "weights": ['uniform', 'distance']},
    "Ensemble": {}
}

def train_model(df, model_name):
    vectorizer_word = TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
        ngram_range=(1, 2)  # Word-level n-grams
        )
    vectorizer_char = TfidfVectorizer(
    analyzer='char_wb',  # Character-level n-grams
    ngram_range=(3, 5),  # 3 to 5 character n-grams
    max_features=5000
    )

    # Combine word and character n-grams
    vectorizer = FeatureUnion([('word', vectorizer_word), ('char', vectorizer_char)])

    X = vectorizer.fit_transform(df['cleaned_comment'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])

    if len(np.unique(y)) < 2:
        print("Not enough variation in data for training.")
        return 0, "Insufficient data", np.array([]), vectorizer

    # Feature selection
    selector = SelectKBest(chi2, k=min(3000, X.shape[1]))
    X = selector.fit_transform(X, y)

    # Resample with ADASYN
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    # Normalize if needed (e.g., for KNN)
    if model_name.startswith("KNN"):
        normalizer = Normalizer()
        X_train = normalizer.fit_transform(X_train.toarray())
        X_test = normalizer.transform(X_test.toarray())

    tuned_param_grid = param_grid.copy()
    if model_name in ["Logistic Regression", "SVM"]:
        tuned_param_grid[model_name]["C"] = [0.01, 0.1, 1, 10]

    # Stratified CV strategy
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = models[model_name]

    grid = GridSearchCV(model, tuned_param_grid[model_name],
                        cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    cm = confusion_matrix(y_test, y_pred)

    # Check for cross-validation performance only if accuracy is 99% or higher
    if acc >= 0.99:
        # Cross-validation score for overfitting check
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
        mean_cv_acc = cv_scores.mean()
        std_cv = cv_scores.std()

        st.subheader(f"📈 Cross-Validation Performance for {model_name} Model")
        st.write(f"**Mean 5-Fold CV Accuracy:** `{mean_cv_acc:.4f}`")
        st.write(f"**Standard Deviation of CV Scores:** `{std_cv:.4f}`")

        if abs(mean_cv_acc - acc) > 0.02:
            st.warning(f"⚠️ Possible overfitting detected. Test Acc `{acc:.4f}` vs CV `{mean_cv_acc:.4f}`")
    else:
        st.write(f"✅ Accuracy is `{acc:.4f}` for {model_name} Model.")


    return acc, report, cm, vectorizer

def generate_wordcloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_data))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


# Main Streamlit Function
def main():
    st.title("📊 **YouTube Comments Analysis**")
    st.markdown("### Analyze video comments and compare models for sentiment analysis using machine learning. 🎥💬")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Single Video Analysis", "Compare Two Videos", "Compare Two Models", "🚨 Abusive Comments"])

    # Initialize session state if not already present
    if 'comments' not in st.session_state:
        st.session_state.comments = None
    if 'video_details' not in st.session_state:
        st.session_state.video_details = None
    if 'sentiment_counts' not in st.session_state:
        st.session_state.sentiment_counts = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}

    # Tab 1: Single Video Analysis
    with tab1:
        st.header("🔍 **Single Video Analysis**")
        api_key = st.text_input("Enter YouTube API Key:", key="api_key_tab1")
        video_id = st.text_input("Enter YouTube Video ID:", key="video_id_tab1")
        model_option = st.selectbox("Select ML Model:", list(models.keys()) + ["All"])
        
        analyze_button = st.button("🔎 Analyze Video")
        
        if analyze_button:
            with st.spinner("Fetching comments..."):
                comments, video_details = fetch_youtube_comments(video_id, api_key)
            
            if not comments:
                st.error("🚨 No comments found for the video ID.")
            else:
                # Store fetched data in session state
                st.session_state.comments = comments
                st.session_state.video_details = video_details
                
                df = prepare_data(comments)
                sentiment_counts = df['sentiment'].value_counts().to_dict()
                st.session_state.sentiment_counts = sentiment_counts  # Store sentiment counts
                
                # Display results
                st.subheader("📋 **All Comments & Classification Table**")
                st.dataframe(df[['comment', 'sentiment']])
                st.subheader(f"### Video: {video_details.get('title', 'Unknown')}")
                st.write(f"👍 Likes: {video_details.get('likes', 0)}  |  👁️ Views: {video_details.get('views', 0)}")
                st.write(f"🧠 Sentiment Analysis:")
                st.write(f"😊 Positive Comments: {sentiment_counts.get('Positive', 0)}")
                st.write(f"😡 Negative Comments: {sentiment_counts.get('Negative', 0)}")
                st.write(f"😐 Neutral Comments: {sentiment_counts.get('Neutral', 0)}")
                
                # Plot sentiment distribution
                st.subheader("📊 Sentiment Distribution")
                fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
                sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), palette=['green', 'red', 'blue'])
                ax.set_xlabel("Sentiment", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.set_title("Sentiment Analysis Results", fontsize=10)
                ax.tick_params(axis='both', labelsize=7)

                for index, value in enumerate(sentiment_counts.values()):
                    ax.text(index, value + 0.5, str(value), ha='center', fontsize=7)
                st.pyplot(fig)

                with st.expander("See Word Cloud for Comments"):
                    generate_wordcloud(df['cleaned_comment'])

                if model_option == "All":
                    results = {model: train_model(df, model)[0] for model in models.keys()}
                    st.session_state.model_results = results  # Store model results
                    st.write("### 📊 Model Accuracy Comparison")
                    st.bar_chart(pd.Series(results, name='Accuracy'))
                else:
                    acc, class_report, conf_matrix, vectorizer = train_model(df, model_option)
                    st.session_state.model_results = {model_option: acc}  # Store individual model results
                    st.write(f"#### **{model_option}** Model")
                    st.markdown(f"<h2 style='color: black;'>📊 Accuracy: {acc * 100:.2f}%</h2>", unsafe_allow_html=True)
                    st.write("### 📝 Classification Report")
                    st.text(class_report)
                    st.write("### 📊 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
                    sns.heatmap(conf_matrix, annot=True, cmap="Blues", ax=ax, fmt="d", annot_kws={"size": 8})
                    ax.set_xlabel("Predicted", fontsize=8)
                    ax.set_ylabel("Actual", fontsize=8)
                    plt.xticks(fontsize=7)
                    plt.yticks(fontsize=7)
                    st.pyplot(fig)
    with tab2:
        st.header("🎥 **Compare Two Videos**")
        api_key = st.text_input("Enter YouTube API Key for Comparison:", key="api_key_tab2")
        video_id_1 = st.text_input("Enter YouTube Video ID 1:", key="video_id1_tab2")
        video_id_2 = st.text_input("Enter YouTube Video ID 2:", key="video_id2_tab2")
        model_option = st.selectbox("Select ML Model for Comparison:", list(models.keys()))

        analyze_button = st.button("🔎 Compare Videos")

        if analyze_button:
            with st.spinner("Fetching comments..."):
                comments_1, video_details_1 = fetch_youtube_comments(video_id_1, api_key)
                comments_2, video_details_2 = fetch_youtube_comments(video_id_2, api_key)

            if not comments_1 or not comments_2:
                st.error("🚨 No comments found for one or both of the video IDs.")
            else:
                # Store comments and video details in session state
                st.session_state.comments_1 = comments_1
                st.session_state.video_details_1 = video_details_1
                st.session_state.comments_2 = comments_2
                st.session_state.video_details_2 = video_details_2

                # Prepare dataframes for both videos
                df_1 = prepare_data(comments_1)
                df_2 = prepare_data(comments_2)

                # Sentiment analysis
                sentiment_counts_1 = df_1['sentiment'].value_counts().to_dict()
                sentiment_counts_2 = df_2['sentiment'].value_counts().to_dict()
                st.session_state.sentiment_counts_1 = sentiment_counts_1
                st.session_state.sentiment_counts_2 = sentiment_counts_2

                # First column: Video 1 analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 **All Comments & Classification Table (Video 1)**")
                    st.dataframe(df_1[['comment', 'sentiment']])
                    st.subheader(f"### Video 1: {video_details_1.get('title', 'Unknown')}")
                    st.write(f"👍 Likes: {video_details_1.get('likes', 0)}  |  👁️ Views: {video_details_1.get('views', 0)}")
                    st.write(f"🧠 Sentiment Analysis")
                    st.write(f"😊 Positive Comments: {sentiment_counts_1.get('Positive', 0)}")
                    st.write(f"😡 Negative Comments: {sentiment_counts_1.get('Negative', 0)}")
                    st.write(f"😐 Neutral Comments: {sentiment_counts_1.get('Neutral', 0)}")
                    st.subheader("📊 Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(3,2))
                    sns.barplot(x=list(sentiment_counts_1.keys()), y=list(sentiment_counts_1.values()), palette=['green', 'red', 'blue'])
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    ax.set_title("Sentiment Analysis Results")

                    # Show values on bars
                    for index, value in enumerate(sentiment_counts_1.values()):
                        ax.text(index, value + 2, str(value), ha='center', fontsize=12)

                    st.pyplot(fig)
                    st.write("### 🧳 Word Cloud")
                    generate_wordcloud(df_1['cleaned_comment'])

                # Second column: Video 2 analysis
                with col2:
                    st.subheader("📋 **All Comments & Classification Table (Video 2)**")
                    st.dataframe(df_2[['comment', 'sentiment']])
                    st.subheader(f"### Video 2: {video_details_2.get('title', 'Unknown')}")
                    st.write(f"👍 Likes: {video_details_2.get('likes', 0)}  |  👁️ Views: {video_details_2.get('views', 0)}")
                    st.write(f"🧠 Sentiment Analysis")
                    st.write(f"😊 Positive Comments: {sentiment_counts_2.get('Positive', 0)}")
                    st.write(f"😡 Negative Comments: {sentiment_counts_2.get('Negative', 0)}")
                    st.write(f"😐 Neutral Comments: {sentiment_counts_2.get('Neutral', 0)}")
                    st.subheader("📊 Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(3,2))
                    sns.barplot(x=list(sentiment_counts_2.keys()), y=list(sentiment_counts_2.values()), palette=['green', 'red', 'blue'])
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    ax.set_title("Sentiment Analysis Results")

                    # Show values on bars
                    for index, value in enumerate(sentiment_counts_2.values()):
                        ax.text(index, value + 2, str(value), ha='center', fontsize=12)

                    st.pyplot(fig)
                    st.write("### 🧳 Word Cloud")
                    generate_wordcloud(df_2['cleaned_comment'])

                # Model comparison section
                col1, col2 = st.columns(2)

                with col1:
                    acc_1, class_report_1, conf_matrix_1, vectorizer_1 = train_model(df_1, model_option)
                    st.session_state.acc_1 = acc_1
                    st.session_state.class_report_1 = class_report_1
                    st.session_state.conf_matrix_1 = conf_matrix_1
                    st.session_state.vectorizer_1 = vectorizer_1

                    st.write(f"#### **{model_option}** Model - Video 1")
                    st.markdown(f"<h2 style='color: black;'>📊 Accuracy: {acc_1 * 100:.2f}%</h2>", unsafe_allow_html=True)
                    st.write("### 📝 Classification Report")
                    st.text(class_report_1)
                    st.write("### 📊 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(3, 2))
                    sns.heatmap(conf_matrix_1, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)

                with col2:
                    acc_2, class_report_2, conf_matrix_2, vectorizer_2 = train_model(df_2, model_option)
                    st.session_state.acc_2 = acc_2
                    st.session_state.class_report_2 = class_report_2
                    st.session_state.conf_matrix_2 = conf_matrix_2
                    st.session_state.vectorizer_2 = vectorizer_2

                    st.write(f"#### **{model_option}** Model - Video 2")
                    st.markdown(f"<h2 style='color: black;'>📊 Accuracy: {acc_2 * 100:.2f}%</h2>", unsafe_allow_html=True)
                    st.write("### 📝 Classification Report")
                    st.text(class_report_2)
                    st.write("### 📊 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(3, 2))
                    sns.heatmap(conf_matrix_2, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)
                                          
    with tab3:
        st.header("🤖 **Compare Two Models**")
        api_key = st.text_input("Enter YouTube API Key for Comparison:", key="api_key_tab3")
        model_option_1 = st.selectbox("Select First ML Model:", list(models.keys()))
        model_option_2 = st.selectbox("Select Second ML Model:", list(models.keys()))
        video_id = st.text_input("Enter YouTube Video ID to Compare Models:", key="video_id_tab3")
        
        analyze_button = st.button("🔎 Compare Models")
        
        if analyze_button:
            with st.spinner("Fetching comments..."):
                comments, video_details = fetch_youtube_comments(video_id, api_key)
            
            if not comments:
                st.error("🚨 No comments found for the video ID.")
            else:
                # Store comments and video details in session state
                st.session_state.comments = comments
                st.session_state.video_details = video_details

                df = prepare_data(comments)
                sentiment_counts = df['sentiment'].value_counts().to_dict()
                st.session_state.sentiment_counts = sentiment_counts

                # First column: Model 1 analysis
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"### 📊 **{model_option_1}** Model")
                    st.write(f"👍 Likes: {video_details.get('likes', 0)}  |  👁️ Views: {video_details.get('views', 0)}")
                    st.write(f"🧠 Sentiment Analysis")
                    st.write(f"😊 Positive Comments: {sentiment_counts.get('Positive', 0)}")
                    st.write(f"😡 Negative Comments: {sentiment_counts.get('Negative', 0)}")
                    st.write(f"😐 Neutral Comments: {sentiment_counts.get('Neutral', 0)}")
                    st.subheader("📊 Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(3, 2))
                    sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), palette=['green', 'red', 'blue'])
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    ax.set_title("Sentiment Analysis Results")

                    # Show values on bars
                    for index, value in enumerate(sentiment_counts.values()):
                        ax.text(index, value + 2, str(value), ha='center', fontsize=12)

                    st.pyplot(fig)

                    # Train and evaluate Model 1
                    acc_1, class_report_1, conf_matrix_1, vectorizer_1 = train_model(df, model_option_1)
                    st.session_state.acc_1 = acc_1
                    st.session_state.class_report_1 = class_report_1
                    st.session_state.conf_matrix_1 = conf_matrix_1
                    st.session_state.vectorizer_1 = vectorizer_1

                    st.markdown(f"<h2 style='color: black;'>📊 Accuracy: {acc_1 * 100:.2f}%</h2>", unsafe_allow_html=True)
                    st.write("### 📝 Classification Report - Model 1")
                    st.text(class_report_1)
                    fig, ax = plt.subplots(figsize=(3, 2))
                    sns.heatmap(conf_matrix_1, annot=True, cmap="Blues", ax=ax)
                    st.pyplot(fig)
                    st.write("### 🧳 Word Cloud")
                    generate_wordcloud(df['cleaned_comment'])

                # Second column: Model 2 analysis
                with col2:
                    st.subheader(f"### 📊 **{model_option_2}** Model")
                    st.write(f"👍 Likes: {video_details.get('likes', 0)}  |  👁️ Views: {video_details.get('views', 0)}")
                    st.write(f"🧠 Sentiment Analysis")
                    st.write(f"😊 Positive Comments: {sentiment_counts.get('Positive', 0)}")
                    st.write(f"😡 Negative Comments: {sentiment_counts.get('Negative', 0)}")
                    st.write(f"😐 Neutral Comments: {sentiment_counts.get('Neutral', 0)}")
                    st.subheader("📊 Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(3, 2))
                    sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), palette=['green', 'red', 'blue'])
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    ax.set_title("Sentiment Analysis Results")

                    # Show values on bars
                    for index, value in enumerate(sentiment_counts.values()):
                        ax.text(index, value + 2, str(value), ha='center', fontsize=12)

                    st.pyplot(fig)

                    # Train and evaluate Model 2
                    acc_2, class_report_2, conf_matrix_2, vectorizer_2 = train_model(df, model_option_2)
                    st.session_state.acc_2 = acc_2
                    st.session_state.class_report_2 = class_report_2
                    st.session_state.conf_matrix_2 = conf_matrix_2
                    st.session_state.vectorizer_2 = vectorizer_2

                    st.markdown(f"<h2 style='color: black;'>📊 Accuracy: {acc_2 * 100:.2f}%</h2>", unsafe_allow_html=True)
                    st.write("### 📝 Classification Report - Model 2")
                    st.text(class_report_2)
                    fig, ax = plt.subplots(figsize=(3, 2))
                    sns.heatmap(conf_matrix_2, annot=True, cmap="Blues", ax=ax)
                    st.pyplot(fig)
                    st.write("### 🧳 Word Cloud")
                    generate_wordcloud(df['cleaned_comment'])

    with tab4:
        api_key = st.text_input("🔑 Enter Your YouTube API Key")
        video_id = st.text_input("🎥 Enter YouTube Video ID")

        if st.button("🔍 Analyze Comments"):
            if not api_key or not video_id:
                st.error("⚠️ Please enter both API Key and Video ID")
            else:
                comments, video_info = fetch_youtube_comments(video_id, api_key)
                
                if comments:
                    # Store the fetched comments and video details in session state
                    st.session_state.comments = comments
                    st.session_state.video_info = video_info
                    
                    st.success(f"✅ Fetched {len(comments)} comments from **{video_info['title']}**")
                    
                    # Process Comments
                    df = prepare_data(comments)
                    st.session_state.df = df

                    # Display Video Stats
                    col1, col2 = st.columns(2)
                    col1.metric("👍 Likes", video_info["likes"])
                    col2.metric("👀 Views", video_info["views"])

                    # Show Processed Comments
                    st.subheader("📋 Processed Comments")
                    st.dataframe(df[["comment", "sentiment", "abusive"]])

                    # Store abusive comments for later reference
                    abusive_comments = df[df["abusive"] == 'offensive']
                    st.session_state.abusive_comments = abusive_comments

                    # Show Abusive Comments Separately
                    if not abusive_comments.empty:
                        st.warning("⚠️ Detected Abusive Comments:")
                        st.dataframe(abusive_comments[["comment", "abusive"]])
                        st.subheader("☁️ Word Cloud of Abusive Comments")
                        generate_wordcloud(abusive_comments["cleaned_comment"])
                else:
                    st.warning("⚠️ No comments found for this video.")

                    
# Run the application
if __name__ == "__main__":
    main()
