from src.logger import logging
from src.pipelines.prediction_pipeline import PredictPipeline
import pandas as pd

from src.pipelines.training_pipeline import trainingPipeline
from src.logger import logging



import os

def run_prediction(featureData: pd.DataFrame) -> int:
    if os.path.exists('artifacts/model.pkl'):
        print(f"The model file file is present in the artifacts directory.")
        logging.info("model.pkl file exist running the prediction now.")

        # now code to call the prediction right away
        predPipeline= PredictPipeline()
        pred= predPipeline.predict(featureData)
        print(pred)
        return pred
            
    else:
        print(f"The model.pkl file is not found in the artifacts directory.")
        logging.info("model.pkl file does not exits, runnint the training pipeline first")
        trainingPipeline()

        # after running the training pipeline the model.pkl file should be present.
        pred= PredictPipeline.predict(featureData)
        print(pred)
        return pred


