
from src.logger import logging
from src.model_components.data_ingestion import DataIngestion
from src.model_components.data_preprocessing import DataPreprocessing
from src.model_components.model_training import ModelTrainer



def  trainingPipeline():
    """
    This function instantiate the training pipeline...
    """
    logging.info("Strating the Model Training Pipeline.")

    dataIngestionObj=DataIngestion()
    train_data,test_data=dataIngestionObj.initiate_dataIngetion()

    data_preprocessing=DataPreprocessing()
    train_arr,test_arr,_=data_preprocessing.initiate_data_preprocessing(train_data,test_data)

    modeltrainer=ModelTrainer()
    print("accuracy of best model is:", modeltrainer.initiate_model_trainer(train_arr,test_arr))

    