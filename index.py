import numpy as np
import string
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download()
nltk.download('punkt')
nltk.download('stopwords')
df = pd.read_csv('spam.csv', encoding='latin-1')
# # Reading Dataset
def load_data(df):
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])
    df.drop_duplicates(keep='first', inplace=True)
    return df

df = load_data(df)

# # Data Transformation and Cleaning
def text_preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert to lower
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # Removing stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    ps = PorterStemmer()
    text = y[:]
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Training
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['text']).toarray()
y = df['target'].values

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=2)

clf = MultinomialNB()
clf.fit(X_train, y_train)

# Streamlit App
st.title("Spam SMS Detection")

# User Input
user_input = st.text_area("Enter a message:")
user_input_transformed = text_preprocess(user_input)
user_input_tfidf = tfidf.transform([user_input_transformed])

# Prediction Button
if st.button("Predict"):
    st.write("Hello")
    prediction = clf.predict(user_input_tfidf)

    if prediction == 0:
        st.success("The message is not spam.")
    else:
        st.error("The message is spam.")
