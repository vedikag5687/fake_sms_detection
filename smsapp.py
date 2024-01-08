import streamlit as st
import pickle
import nltk 
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
ps = PorterStemmer()

nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    filtered_tokens = []

    stop_words = set(stopwords.words('english'))

    for token in tokens:
        if token.isalnum() and token not in stop_words and token not in string.punctuation:
            filtered_tokens.append(ps.stem(token))

    return " ".join(filtered_tokens)
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter Your Message")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
       st.header("Spam")

    else:
       st.header("Not Spam")