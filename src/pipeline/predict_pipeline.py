import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object,load_json

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            threshold_path = os.path.join("artifacts","threshold.json")
            threshold = load_json(threshold_path)['threshold']
            data_scaled = preprocessor.transform(features)
            proba_pred = model.decision_function(data_scaled)
            predictions_inference = (proba_pred > threshold).astype(int)
            return predictions_inference
        except Exception as e:
            raise CustomException(e,sys)
        


