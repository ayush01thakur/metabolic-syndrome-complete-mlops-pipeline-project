import os
import sys

from src.logger import logging
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Saves the objects (files and folder)
    args: 
        filer_path: path of the file
        obj: the object that needed to be saved at the given path
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("error occured while saving the object", obj)
        raise e
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    """
    This function trains->tests->evaluates the listed models in model_training file.
    returns:
            report (evaluation report in format: {modelName: accuracy})
    """
    try:
        report = {}
        logging.info("Training over all the listed models and testing for the accuracy score to generate report")

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info("error occured while evaluating listed models")
        raise e
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("error occured while loading the object- ", file_path)
        raise e
    
