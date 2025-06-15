# 📰 Fake News Detection using Machine Learning

This project demonstrates a machine learning approach to detect **fake news articles** using text classification. We use **TF-IDF vectorization** and **Logistic Regression** on a dataset containing article metadata such as title, text, author, category, and source.

---

## 📌 Overview

The primary goal is to classify news articles as either **Real** or **Fake** based on textual content and metadata. This helps reduce misinformation and builds tools for media credibility.

---

## 🧠 Model & Features

### ✅ Features Used:
- `title`: Headline of the article
- `text`: Main article body
- `author`: Name of the journalist
- `category`: Topic/section of the article (Politics, Business, etc.)
- `source`: Media outlet (e.g., CNN, NY Times)

These fields are combined and preprocessed into a single text feature.

### 🧪 Model Used:
- **TF-IDF Vectorizer** for converting text into numerical features
- **Logistic Regression** for binary classification (`real` vs `fake`)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn (ML pipeline)
- Jupyter Notebook

---

## ⚙️ How It Works

1. **Preprocessing**
   - Fill missing values with empty strings
   - Merge `author`, `title`, `text`, `category` into a single `content` feature
   - Clean text (remove special characters, lowercase, stem, remove stopwords)

2. **Vectorization**
   - Use `TfidfVectorizer` to convert text into feature vectors

3. **Model Training**
   - Train/test split (80-20) with label stratification
   - Train Logistic Regression model

4. **Evaluation**
   - Accuracy score
   - Test with manual user input


