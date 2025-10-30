# ✈️ Airline Tweet Sentiment Dashboard & Analysis

This project applies Natural Language Processing (NLP) to classify sentiment in airline-related tweets. Using a real-world dataset, we build a reproducible pipeline that transforms noisy social media text into actionable insights.

## 🔍 Project Overview

- **Goal**: Predict tweet sentiment (positive, neutral, negative) using text-based features  
- **Dataset**: Public airline tweet dataset with labeled sentiment  
- **Tech Stack**: Python, Pandas, Scikit-learn, Seaborn, Matplotlib, TF-IDF
  
## 📊 Streamlit Dashboard Features

This project includes an interactive dashboard built with Streamlit to showcase the sentiment analysis pipeline in action.

- Upload CSVs of airline tweets
- Clean and preprocess text using NLP
- Visualize sentiment distribution
- Filter tweets by sentiment and tweet length
- Train ML models (Logistic Regression or Random Forest)
- View model performance (accuracy, classification report, confusion matrix)
- Predict sentiment on new tweets
- Download predictions as CSV

## 🚀 Live App

Explore the deployed dashboard here:  
👉 [Launch the Sentiment Dashboard](https://share.streamlit.io/slayerranjan/social-media-sentiment-analysis/main/app.py)


## 🧪 Methodology

1. **Data Cleaning**: Regex-based removal of URLs, mentions, punctuation, and numbers  
2. **Feature Engineering**: Tweet length, word count, and TF-IDF vectorization  
3. **Modeling**: Logistic Regression and Random Forest comparison  
4. **Evaluation**: Accuracy, classification report, confusion matrix, cross-validation

📊 Results
---

**Logistic Regression**
- Accuracy: **81%** on test set  
- Cross-validation Accuracy: **~79.2%** (5-fold)  
- F1-score: **0.88** (negative), **0.61** (neutral), **0.71** (positive)  
- Confusion matrix shows high precision for negative sentiment

**Random Forest**
- Accuracy: **~76%**  
- Precision: Strong for negative sentiment  
- Recall: Weak for neutral sentiment  
- F1-score: Lower than Logistic Regression overall

**Insights**
- Neutral tweets are hardest to classify due to vocabulary overlap  
- Logistic Regression consistently outperforms Random Forest on this dataset  
- Ensemble models like Random Forest offer robustness but struggle with class ambiguity in text


## 📁 Repo Structure

├── sentiment_analysis_notebook.ipynb 
├── twitter_airline_sentiment.csv
├── README.md


## 🚀 Highlights

- Clean, well-commented notebook with numbered markdown sections  
- Visualizations for sentiment distribution, tweet length, and model diagnostics  
- Reproducible pipeline ready for deployment or extension

## 📓 View the Notebook

Explore the full sentiment analysis pipeline in the Jupyter notebook:  
👉 [Social_Media_Analysis.ipynb](https://github.com/slayerranjan/Social-media-sentiment-analysis/blob/main/Social_Media_Analysis.ipynb)



## 📌 Author

**Ranjan** — Data enthusiast with a passion for clarity, storytelling, and real-world impact.  
Connect on [LinkedIn](https://www.linkedin.com/in/ranjan-shettigar-b89808309) or explore more projects on [GitHub](https://github.com/slayerranjan).

