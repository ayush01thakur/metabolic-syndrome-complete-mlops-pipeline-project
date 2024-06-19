import os
import pandas as pd
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        This function perfroms the prediction on the data entered by the user.
        args:
            features= dictionary of feature and its values
        returns:
            predicted value (in this case classification)
        """
        logging.info("Running the predict function for prediction")
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            
            logging.info("loading model.pkl and preprocessor.pkl")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            logging.info("loading completed")
        

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)

            logging.info("prediction Completed; returning the predicted value")
            return preds
        
        except Exception as e:
            logging.info("Error in predict function in prediction pipeline")
            raise e


def get_data_as_data_frame():
    """
    This function interacts with the web page and dets the user data, then convert the data into dataframe
    
    """
    try:
        logging.info("converting the dictionary data into the required dataframe object")
        data_inputs= {'Age':10, 'Sex':'Male', 'Marital':'Unmarried', 'Income': 500000, 'Race': 'White', 'WaistCirc':30, 'BMI':26,
       'Albuminuria':3.4, 'UrAlbCr':44, 'UricAcid':3.4, 'BloodGlucose':23, 'HDL':33,
       'Triglycerides':12}
        
        return pd.DataFrame(data_inputs)

    except Exception as e:
        logging.info("Error in get_data_as_data_frame function")
        raise e