import pickle
import numpy as np
import pandas as pd
from flask import Flask,jsonify,render_template,request
from sklearn.preprocessing import StandardScaler


application=Flask(__name__)
app=application


## importing ridge regression and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST','GET'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')
def hello_world():
    return render_template('index.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)