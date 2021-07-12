import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
model_showup = pickle.load(open('model_showup.pkl', 'rb'))
model_activity = pickle.load(open('model_activity.pkl', 'rb'))
cleaned_data = {}
test_data = {}

def clean_data(test_data):
    print('cleaning data')
    data = test_data.copy()

    data.drop(columns=['id'], inplace=True)

    data.age.fillna(data.age.mean(), inplace=True)
    data.gender.fillna(0, inplace=True)
    data.hobbies_sports.fillna(0, inplace=True)
    data.hobbies_environment.fillna(0, inplace=True)
    data.hobbies_fitness.fillna(0, inplace=True)
    data.hobbies_cooking.fillna(0, inplace=True)
    data['differently abled'] = pd.Categorical(data['differently abled']).codes
    
    for col in ['age', 'gender', 'hobbies_sports', 'hobbies_environment','hobbies_cooking','hobbies_fitness']:
        data[col] = data[col].astype('int64')

    return data

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict_showup',methods=['POST'])
def predict_showup():
    print('inside predict showup')
    file = request.files['fileToUpload']  
    test_data = pd.read_csv(file)
    cleaned_data = clean_data(test_data)
    prediction = model_showup.predict_proba(cleaned_data)

    out_data = test_data.copy()
    out_data['show_prob'] = prediction[:,1]
    final_out_data = out_data[['id', 'show_prob']]
    final_out_data.to_csv("ShowUpPrediction.csv", index=False)
    return jsonify("Results are downloaded in downloads folder with the name ShowUpPrediction.csv")

@app.route('/predict_activity',methods=['POST'])
def predict_activity():
    print('inside predict activity')
    file = request.files['fileToUploadActivity']  
    test_data = pd.read_csv(file)
    cleaned_data = clean_data(test_data)
    print('data cleaned')
    prediction = model_activity.predict_proba(cleaned_data)

    out_data = test_data.copy()
    out_data[['on-field', 'spot', 'moving', 'social']] = prediction
    final_out_data = out_data[['id', 'on-field', 'spot', 'moving', 'social']]
    final_out_data.to_csv(os.path.expanduser("~") + "/Downloads/ActivityPrediction.csv", index=False)
    return jsonify("Results are downloaded in downloads folder with the name ActivityPrediction.csv")

if __name__ == "__main__":
    app.run(debug=True)
