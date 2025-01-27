import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import nltk

nltk.data.path.append("C:/Users/user-admin/Desktop/Spam Email Detetion/venv/Lib/site-packages/nltk/tokenize/punkt.py")
logging.getLogger('nltk').setLevel(logging.CRITICAL)
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    if isinstance(text,list):
        text = "".join([str(item) for sublist in text for item in (sublist if isinstance(sublist,list) else [sublist])])
    words = word_tokenize(text.lower())
    #print(type(text))
    filtered_word = [ps.stem(w) for w in words if w not in stop_words and w.isalnum()]
    return " ".join(filtered_word)  
    
data = pd.read_csv('spam.csv',encoding = 'latin-1')
data = data[['v1','v2']]
data.columns = ['label','text']

data['label'] = data['label'].map({'spam':1,'ham':0})

data['text'] = data['text'].apply(preprocess_text)

x = data['text']
y = data['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_train_vectorized,y_train)

y_pred = model.predict(x_test_vectorized)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))

def predict_spam(email_text):
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] ==1 else "Not Spam"

text_email_1 = "Congratulations! You've won a $1000 gift card. Click here to claim now"
text_email_2 = "Hi John, can we reschedule our meeting to tomorrow morning?"

print(f"Text Email 1:{predict_spam(text_email_1)}")
print(f"Text Email 2:{predict_spam(text_email_2)}")

app = FastAPI()

class Prediction_input(BaseModel):
    mail: str
    
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins =["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)
@app.get("/")
def read_root():
    return {"message":"Service is lived"}

@app.post("/predict")
def predict(input_data:Prediction_input):
    features = [[
        input_data.mail
    ]]
    prediction = predict_spam(features)
    return {"prediction": str(prediction)}
