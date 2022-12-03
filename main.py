from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
app= Flask(__name__)
data=pd.read_csv('data.csv')
pipe= pickle.load(open('HousePrices.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    CRM= request.form.get('CRM')
    ZN= request.form.get('ZN')
    INDUS= request.form.get('INDUS')
    CHAS= request.form.get('CHAS')
    NOX= request.form.get('NOX')
    RM= request.form.get('RM')
    AGE= request.form.get('AGE')
    DIS= request.form.get('DIS')
    RAD= request.form.get('RAD')
    TAX= request.form.get('TAX')
    PTR= request.form.get('PTR')
    # B= request.form.get('B')
    LSTAT= request.form.get('LSTAT')
    print("Got the values",CRM, LSTAT)
    input=pd.DataFrame([[CRM, ZN, INDUS , CHAS,NOX, RM,AGE,DIS,RAD,TAX,PTR, LSTAT]],columns=['CRM', 'ZN','INDUS', 'CHAS','NOX', 'RM','AGE','DIS','RAD','TAX','PTR', 'LSTAT'])
    prediction=pipe.predict(input)[0]
    prediction=prediction*1000
    return str(np.round(prediction,3))

if __name__=="__main__":
    app.run(host='0.0.0.0')
