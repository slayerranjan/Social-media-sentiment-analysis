# 1. Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# 2. Streamlit Page Config
st.set_page_config(page_title="Airline Tweet Sentiment Dashboard", layout="wide")
st.title("✈️ Airline Tweet Sentiment Analysis")
st.markdown("Analyze and predict sentiment from airline-related tweets using NLP and ML.")
st.write("Current working directory:", os.getcwd())

# 3. Define Cleaning Function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

# 4. Define Prediction Function
def predict_sentiment(df, model, vectorizer):
    df['clean_text'] = df['text'].apply(clean_text)
    X = vectorizer.transform(df['clean_text']).toarray()
    df['Predicted Sentiment'] = model.predict(X)
    return df[['text', 'Predicted Sentiment']]

# 5. Upload Dataset
uploaded_file = st.file_uploader("Upload Airline Tweet CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 6. Data Cleaning
    df['clean_text'] = df['text'].apply(clean_text)
    df['tweet_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

    # 7. Show Raw and Cleaned Data
    with st.expander("📂 Show Raw and Cleaned Data"):
        if st.checkbox("Show raw and cleaned data"):
            st.write(df[['text', 'clean_text', 'airline_sentiment']].head())

    # 8. Sidebar Filters
    st.sidebar.header("🔍 Filter Tweets")
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", options=["All"] + df['airline_sentiment'].unique().tolist())
    min_length, max_length = st.sidebar.slider("Tweet Length Range", 0, 300, (0, 300))

    filtered_df = df.copy()
    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df['airline_sentiment'] == selected_sentiment]
    filtered_df = filtered_df[(filtered_df['tweet_length'] >= min_length) & (filtered_df['tweet_length'] <= max_length)]

    # 9. Sentiment Distribution Chart
    st.subheader("📈 Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='airline_sentiment', palette='pastel')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    st.pyplot(fig)

    # 10. Model Selection + Training
    st.sidebar.header("⚙️ Model Settings")
    model_choice = st.sidebar.selectbox("Choose ML Model", ["Logistic Regression", "Random Forest"])

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['clean_text']).toarray()
    y = df['airline_sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 11. Tabs for Filtered Tweets and Model Performance
    tab1, tab2 = st.tabs(["📄 Filtered Tweets", "📊 Model Performance"])

    with tab1:
        st.write(filtered_df[['text', 'airline_sentiment']].head())

    with tab2:
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**{model_choice} Accuracy:** {accuracy:.2%}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    # 12. Predict Sentiment from Uploaded Tweets
    st.subheader("📁 Predict Sentiment from Uploaded Tweets")
    custom_file = st.file_uploader("Upload your own tweet CSV for prediction", type=["csv"], key="custom")
    if custom_file:
        custom_df = pd.read_csv(custom_file)
        predicted_df = predict_sentiment(custom_df, model, tfidf)
        st.write(predicted_df.head())

        # ✅ Download Button
        st.download_button(
            label="Download Predictions as CSV",
            data=predicted_df.to_csv(index=False),
            file_name="predicted_sentiments.csv",
            mime="text/csv"
        )

else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()
