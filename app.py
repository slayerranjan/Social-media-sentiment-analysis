# 1. Import Libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Add Streamlit Page Config

st.set_page_config(page_title="Airline Tweet Sentiment Dashboard", layout="wide")
st.title("✈️ Airline Tweet Sentiment Analysis")
st.markdown("Analyze and predict sentiment from airline-related tweets using NLP and ML.")



# 3. Upload Dataset

uploaded_file = st.file_uploader("Upload Airline Tweet CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("twitter_airline_sentiment.csv")  # fallback to default                            


# 4. Add Data Cleaning Logic

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['tweet_length'] = df['clean_text'].apply(len)
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

# 5. Show Raw and Cleaned Data

if st.checkbox("Show raw and cleaned data"):
    st.write(df[['text', 'clean_text', 'airline_sentiment']].head())


# 🔍 Sidebar Filters
st.sidebar.header("🔍 Filter Tweets")
selected_sentiment = st.sidebar.selectbox("Select Sentiment", options=["All"] + df['airline_sentiment'].unique().tolist())
min_length, max_length = st.sidebar.slider("Tweet Length Range", 0, 300, (0, 300))

filtered_df = df.copy()
if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df['airline_sentiment'] == selected_sentiment]
filtered_df = filtered_df[(filtered_df['tweet_length'] >= min_length) & (filtered_df['tweet_length'] <= max_length)]

st.subheader("📄 Filtered Tweets")
st.write(filtered_df[['text', 'airline_sentiment']].head())


# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['airline_sentiment']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict
y_pred = lr_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.subheader("📊 Model Performance")
st.write(f"**Logistic Regression Accuracy:** {accuracy:.2%}")


# Classification Report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lr_model.classes_, yticklabels=lr_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)



# 📁 Predict Sentiment from Uploaded Tweets
st.subheader("📁 Predict Sentiment from Uploaded Tweets")
custom_file = st.file_uploader("Upload your own tweet CSV for prediction", type=["csv"], key="custom")
if custom_file:
    custom_df = pd.read_csv(custom_file)
    custom_df['clean_text'] = custom_df['text'].apply(clean_text)
    custom_X = tfidf.transform(custom_df['clean_text']).toarray()
    custom_preds = lr_model.predict(custom_X)
    custom_df['Predicted Sentiment'] = custom_preds
    st.write(custom_df[['text', 'Predicted Sentiment']].head())


