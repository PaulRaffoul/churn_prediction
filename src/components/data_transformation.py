import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

    def cleaner(self,df):
        """
        Function that does the basic cleaning of a dataset like filling missing values, type conversions etc.
        """
        try:
            df.drop(columns={'Country','State','City','CustomerID','CLTV','Zip Code','Latitude','Longitude','Lat Long','Count','Churn Label','Churn Reason'},inplace=True)
            df.loc[df['Total Charges']==' ','Total Charges'] = 0
            df['Total Charges'] = df['Total Charges'].astype(float)
            return df
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_transformer_object(self):
        pass

    def initiate_data_transformation(self,train_path,test_path):
        pass

