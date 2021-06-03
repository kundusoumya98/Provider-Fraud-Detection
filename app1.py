from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
import re
import pickle


import flask

app = Flask(__name__)


def metadataset(X):
    yhattest=[]
    for i in range(10):
        t=i%6
        if t==0:
            filename = 'basemodel1.sav'
            model= pickle.load(open(filename, 'rb'))
            ytest=model.predict(X)
        elif t==1:
            filename = 'basemodel2.sav'
            model= pickle.load(open(filename, 'rb'))
            ytest=model.predict(X)
        elif t==2:
            filename = 'basemodel3.sav'
            model= pickle.load(open(filename, 'rb'))
            ytest=model.predict(X)
        elif t==3:
            filename = 'basemodel4.sav'
            model= pickle.load(open(filename, 'rb'))
            ytest=model.predict(X)
        elif t==4:
            filename = 'basemodel5.sav'
            model= pickle.load(open(filename, 'rb'))
            ytest=model.predict(X)
        else:
            filename = 'basemodel6.sav'
            model= pickle.load(open(filename, 'rb'))
            ytest=model.predict(X)
        yhattest.append(ytest)
    return(np.transpose(yhattest))

def findcorrspondingvector(id):
    data= pd.read_csv('newfile.csv')
    data=data.drop(['PotentialFraud','Unnamed: 0'], axis = 1)
    print(id)
    row = data[data['Provider']==id].copy()
    #row=row.drop(["Provider"],axis=1)
    print(row)
    return(row)

def remoovespace(id):
    return id.replace(" ", "")
      
        
        

        
@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    review_text=request.form['review_text']
    print(review_text)
    review_text=str(review_text)
    review_text=remoovespace(review_text)
    data=findcorrspondingvector(review_text)
    if data.empty==True:
        return jsonify('data is not present enter the correct id ')
    
    else:
        print('**********************************')
        print(data)
        data=data.drop(["Provider"],axis=1)
        testmeta=metadataset(data)
        filename = 'metalogistic_model.sav'
        bestmeta= pickle.load(open(filename, 'rb'))
        prediction=bestmeta.predict(testmeta)
    
        if prediction==1:
            prediction = "Fraud provider don't invest your money"
        else:
            prediction = "not a fraud invest your money"

    return jsonify('Provider is ', prediction)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



