import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

class DataIngestion:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts','train.xlsx')
        self.test_data_path = os.path.join('artifacts','test.xlsx')
        self.raw_data_path = os.path.join('artifacts','data.xlsx')

    def initiate_data_ingestion(self):
        logging.info('enter the data ingestion component')
        try:
            df = pd.read_excel('data\Telco_customer_churn.xlsx')
            logging.info('Read the data')

            #create the train data folder
            os.makedirs(os.path.dirname(self.train_data_path),exist_ok=True)

            df.to_excel(self.raw_data_path, index=False,header=True)

            logging.info('Train test split started')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42,stratify=df['Churn Value'])

            train_set.to_excel(self.train_data_path, index=False,header=True)
            test_set.to_excel(self.test_data_path, index=False,header=True)

            logging.info('Data split ended')

            return(
                self.train_data_path,
                self.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)




    
