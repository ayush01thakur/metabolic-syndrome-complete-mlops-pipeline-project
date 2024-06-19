from src.pipelines.training_pipeline import trainingPipeline

if __name__=="__main__":
    # running the training pipeline... once i run this I'll will have the model.pkl and preprocessing.pkl 
    # files so later which will help in the prediction pipeline.
    
    trainingPipeline()