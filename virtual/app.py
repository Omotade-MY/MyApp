# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:50:36 2022

@author: Omotade
"""
from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
# Initalise the Flask app
app = Flask(__name__)

with open('model_deploy-v1.mdl', 'rb') as mdl:
    
    model = pickle.load(mdl)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(float, feature_list))
    final_features = np.array(feature_list).reshape(1, 12) 
    
    prediction = model.predict(final_features)
    output = float(prediction[0])
    """if output == 1:
        text = ">50K"
    else:
        text = "<=50K"
    """

    return render_template('index.html', prediction_text='House Price is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    
