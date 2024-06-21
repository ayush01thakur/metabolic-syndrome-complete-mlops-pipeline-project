from flask import Flask,request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import PredictPipeline, convertToDataframe
from src.logger import logging

application = Flask(__name__, template_folder='templates')

app=application

# code for adjusting the values mapping with model requirements and user input
def settleData(data: dict)-> dict:
    logging.info("changing the values/units of waist circ and Albuminuria")
    # setting the inr value according to dataset
    try:
        data['WaistCirc']= [round(float(data['WaistCirc'][0]) * 2.54, 1)]

        # transforming the Albuminuria values
        if(data['Albuminuria'][0]=='Normal'):
            data['Albuminuria']=[0]
        elif data['Albuminuria'][0]=='Medium':
            data['Albuminuria']=[1]
        else:
            data['Albuminuria']=[2]

        logging.info("transformation into those units done;")
        return data
    
    except Exception as e:
        logging.info("Error occured while settling data... ")
        raise e



## Route for a index page

@app.route('/',methods= ['GET','POST'])
def predict_datapoint():
    print("inside the predict_datapoint()")

    if request.method=='GET':
        logging.info(" request.method=GET called, redenring website index.html ")
        return render_template('index.html')
    else:
        try:
            logging.info("getting data from the form ")
            # print(request.form)
            logging.info(f"Form data: {request.form}")

            data= {'Age': [int(request.form.get('age'))],
                'Sex': [request.form.get('sex')],
                'Marital': [request.form.get('marital')],
                'Race': [request.form.get('race')],
                'WaistCirc':[float(request.form.get('waistCirc'))],
                'BMI': [float(request.form.get('bmi'))],
                'Albuminuria':[request.form.get('albuminuria')],
                'UrAlbCr':[float(request.form.get('urAlbCr'))],
                'UricAcid':[float(request.form.get('uricAcid'))],
                'BloodGlucose':[float(request.form.get('bloodGlucose'))],
                'HDL':[float(request.form.get('hdl'))],
                'Triglycerides':[float(request.form.get('triglycerides'))]
                }
            
            logging.info("mapping data to the required data for the preprocessing")
            data= settleData(data)
            print(data)

            logging.info("Converting data into dataframe for preprocessing")        
            pred_df=convertToDataframe(data)

            logging.info("Initiating the prediction pipeline")
            predict_pipeline=PredictPipeline()
            results=predict_pipeline.predict(pred_df)

            logging.info(f"storing the result in results variable {results}")

            return render_template('index.html', results=results[0])
            

        except Exception as e:
            logging.info(f"Error processing form data: {str(e)}")
            raise e
    

if __name__=="__main__":
    app.config['EXPLAIN_TEMPLATE_LOADING'] = True
    app.run(host="0.0.0.0")        

