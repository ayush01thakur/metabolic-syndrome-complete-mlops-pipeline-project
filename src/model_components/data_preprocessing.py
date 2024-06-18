
import os
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.logger import logging
from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")



class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config= DataPreprocessingConfig()

    def get_data_preprocessing_object(self):
        """
        This function handles all the preprocessing pipeline for the data
        """

        try:
            # numerical and categorical columns
            logging.info("listing numericals and categoricals columns")
            data= pd.read_csv('raw_data\Metabolic Syndrome.csv')
            data = data=data.drop(columns=['seqn'])

            numeric_columns= data.select_dtypes(include='number').columns.drop("MetabolicSyndrome")
            categorical_columns= data.select_dtypes(include='object').columns

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numeric_columns}")

            logging.info("constructing pipelines for numeric and categorical columns")

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info("numerical and categorical columns pipeline created.")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numeric_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("Some error while making preprocessing the data")
            raise e
        

    def initiate_data_preprocessing(self,train_path,test_path):
        """
        Now with the preprocessing object this function will inititate the preprocessing on the data;
        """
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_preprocessing_object()

            target_column_name="MetabolicSyndrome"
            numerical_columns = train_df.select_dtypes(include='number').columns
            categorical_columns= train_df.select_dtypes(include='object').columns

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_preprocessing_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_preprocessing_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Error in data preprocessing initiation")
            raise e
        
        