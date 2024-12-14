import streamlit as st
import joblib
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # Fixed: Added parentheses
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Fixed: Correct PorterStemmer instance

    return " ".join(y)

# Streamlit app title
st.title("SMS-Detection Classifier")

# Load vectorizer and model using joblib
tfidf = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# Input text box
input_sms = st.text_area("Enter the SMS")

if st.button('Predict'):
    # Preprocess input
    transformed_sms = transform_text(input_sms)

    # Vectorize input
    vector_input = tfidf.transform([transformed_sms])  # Fixed: Pass as a list

    # Predict
    result = model.predict(vector_input)[0]

    # Output
    if result == 0:
        st.header("NOT SPAM")
    else:
        st.header("SPAM")
