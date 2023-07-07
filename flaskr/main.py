import pickle
from flask import Flask
from .model import predict # Import predict function from model.py
from transformers import RoFormerTokenizer,RoFormerForSequenceClassification
import torch

'''
Initiate a new flaskr app
1. Input some random secret key to be used by the application 
2. Input some flaskr commands that would be used by the application
'''
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='\xe0\xcd\xac#\x06\xd9\xe4\x00\xa5\xf2\x88\xc3\xef$\xa5\x05n\x97\xd8\x1269i\xd3'
)
from flask import (
    redirect, render_template, request, session, url_for
)


'''
Load the machine learning libraries 
1. Logistic regression model is used to predict the sentiment on the newly computed matrix
'''

# Load the machine learning model
tokenizer = RoFormerTokenizer.from_pretrained('./flaskr/static/Roformer/')
model = RoFormerForSequenceClassification.from_pretrained('./flaskr/static/Roformer/')
print(predict(model=model,tokenizer = tokenizer, text='I love you'))

'''
Home Page
1. It will take both GET and POST requests 
2. For GET request, base.html (homepage) will be rendered without any results shown
3. For POST request, input message will be obtained from the form in base.html.
    a) Session will then be cleared (to remove anything belonged to previous session) and 'message' will be passed into the session 
    so that it can be reused throughout the session
    b) The page will then be redirected to /result page
'''
@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        message = request.form['message']
        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))
    return render_template("base.html")


'''
Result Page
1. It will take both GET and POST requests 
2. For GET request, 'message' will be obtained from the session, remember the 'message' is from the Home page! 
    a) Sentiment and its score(probability) will be predicted by passing in the vectorizer (optional), model and message from the session
    b) The result page will then be rendered based on the message, sentiment and score computed by the predictions
3. For POST request, input message will be obtained from the form in result.html 
    a) Session will then be cleared (to remove anything belonged to previous session) and 'message' will be passed into the session 
    so that it can be reused throughout the session
    c) The page will then be redirected to /result page
'''
@app.route('/result', methods=('GET', 'POST'))
def result():
    message = session.get('message')
    df_pred = predict(model=model,tokenizer = tokenizer, text=message)
    sentiment = df_pred.head(1)['sentiment'].values[0]
    Nscore = df_pred.head(1)['Nscore'].values[0]
    Pscore = df_pred.head(1)['Pscore'].values[0]
    if request.method == 'POST':
        message = request.form['message']
        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))
    return render_template("result.html", message=message, sentiment=sentiment, Nscore=Nscore,Pscore=Pscore)

app.run()