import streamlit as st
import pickle
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
if st.button('Predict'):
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
            
        text = y[:]
        y.clear()
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
            
        text = y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
        return " ".join(y)
    tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
    st.title("Email/SMS Spam classifier")

    input_text = st.text_input("Enter the message")

    #preprocessing
    transformed_sms = transform_text(input_text)
#vectorize
    vector_input = tfidf.transform([transformed_sms])
#predict
    result = model.predict(vector_input)[0]
#Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
    