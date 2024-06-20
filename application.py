from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import PredictPipeline, get_data_as_data_frame
from src.utils import settleData

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data= {'Age': request.form.get('age'),
               'Sex': request.form.get('sex'),
               'Marital': request.form.get('marital'),
               'Income': request.form.get('income'),
               'Race': request.form.get('race'),
               'WaistCirc':request.form.get('waistCirc'),
               'BMI': request.form.get('bmi'),
               'Albuminuria':request.form.get('albuminuria'),
               'UrAlbCr':request.form.get('urAlbCr'),
               'UricAcid':request.form.get('uricAcid'),
               'BloodGlucose':request.form.get('bloodGlucose'),
               'HDL':request.form.get('hdl'),
               'Triglycerides':request.form.get('triglycerides')
               }
        
        data= settleData(data)

        
        pred_df=get_data_as_data_frame(data)
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        return render_template('index.html', results=results[0])
    

if __name__=="__main__":

    app.run(host="0.0.0.0")        

