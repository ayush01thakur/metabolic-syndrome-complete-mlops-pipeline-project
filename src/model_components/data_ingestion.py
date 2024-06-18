import os
import sys
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.model_components.data_preprocessing import DataPreprocessing, DataPreprocessingConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_dataIngetion(self):
        """
        Initiates the data ingestion
        """
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv('raw_data\Metabolic Syndrome.csv')
            df=df.drop(columns=['seqn'])

            logging.info('Read the dataset (metabolic Syndrome.csv)')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error while initiation of data ingestion: {e}")
            raise e

if __name__=="__main__":
    dataIngestionObj=DataIngestion()
    train_data,test_data=dataIngestionObj.initiate_dataIngetion()

    data_preprocessing=DataPreprocessing()
    train_arr,test_arr,_=data_preprocessing.initiate_data_preprocessing(train_data,test_data)

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))