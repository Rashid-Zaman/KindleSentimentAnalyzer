from flask import Flask, request, render_template
import pickle
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from bs4 import BeautifulSoup

app = Flask(__name__)


## Load tfidfvectorizer and model
model=pickle.load(open('multinomial_classifier.pkl','rb'))
tfidf=pickle.load(open('TfidfVectorizer.pkl','rb'))


def preprocess(message):
    message=message.lower()
    message=re.sub('[^a-zA-Z0-9 ]','',message)
    message=' '.join([word for word in message.split() if word not in stopwords.words('english')] )
    message=re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','',str(message))
    message=BeautifulSoup(message,'lxml').get_text()
    message=' '.join(message.split())
    
    input_array=tfidf.transform([message])
    return input_array
    
    
    


@app.route('/',methods=['GET','POST'])
def analyze_sentiment():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        message=request.form['message']
        input_array=preprocess(message)
        prediction = model.predict(input_array)[0]
        result="Positive Review" if prediction == 1 else "Negative Review"
        
        return render_template('home.html',result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=9000,debug=True)
