import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()


    def initiate_model_trainer(self,train_array,test_array):
        """
        This function inititates the model training process by evaluating several models
        returns: 
                accuracy of the best model
                also creates model.pkl file in artifacts folder
        """
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], train_array[:,-1],
                test_array[:,:-1],  test_array[:,-1]
            )

            models = {
                "Random Forest Classifier": RandomForestClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier()
                # "KNeighbors Classifier": KNeighborsClassifier()
            }

            # Hyperparameter tuning;
            logging.info("initializing the params for hyperparameter tuning for respective models")
            params={
                "Decision Tree Classifier": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 20],
                    'min_impurity_decrease': [0.0, 0.1, 0.2]
                },
                "Random Forest Classifier":{
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20, 50],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Gradient Boosting Classifier":{
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.5, 0.7, 0.9, 1.0],
                    'max_depth': [3, 5, 7, 10],
                    'n_estimators': [100, 200, 500]
                },
                
                "XGBClassifier":{
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'max_depth': [3, 5, 7, 10],
                    'n_estimators': [100, 200, 500]
                },
        
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01, 0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.8:
                logging.info("Accuracy of the best model is bad; acc<0.8")

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc = accuracy_score(y_test, predicted)
            return acc
            

            
        except Exception as e:
            logging.info("Error in initiate_model_trainer...")
            raise e