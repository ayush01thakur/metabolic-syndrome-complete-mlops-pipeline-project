from src.logger import logging
from src.pipelines.prediction_pipeline import PredictPipeline
from src.pipelines.prediction_pipeline import get_data_as_data_frame
import pandas as pd

def run_prediction(featureData: pd.DataFrame):
    pred= PredictPipeline.predict(featureData)
    print(pred)

if __name__=="__main__":
    # here write the code to get the data for prediction 
    featureData= get_data_as_data_frame()
    run_prediction(featureData)
