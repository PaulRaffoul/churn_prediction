import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import map_yes_no_to_binary,map_gender,save_object
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler,LabelEncoder,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

    def cleaner(self,df):
        """
        Function that does the basic cleaning of a dataset like filling missing values, type conversions etc.
        """
        try:
            df.drop(columns={'Country','State','City','CustomerID','CLTV','Zip Code','Latitude','Longitude','Lat Long','Count','Churn Label','Churn Reason','Churn Score'},inplace=True)
            df.loc[df['Total Charges']==' ','Total Charges'] = 0
            df['Total Charges'] = df['Total Charges'].astype(float)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def get_transformer_object(self):

        try:
            #define columns
            num_columns = ['Monthly Charges','Total Charges','Tenure Months'] #numerical
            categ_columns_one = ['Multiple Lines','Internet Service','Online Security','Online Backup','Device Protection','Tech Support','Streaming TV',
                                'Streaming Movies','Contract','Payment Method',] #one hot encoding
            categ_columns_two = ['Phone Service','Paperless Billing','Partner','Dependents','Senior Citizen'] #label encoding
            categ_columns_three = ['Gender'] #df["gender"] = df["gender"].map({"Female":1, "Male":0})

            #initiate transformers
            num_pipeline= Pipeline(steps=[("scaler",MinMaxScaler())])
            cat_pipeline_one = Pipeline(steps=[("one_hot_encoder",OneHotEncoder())])
            cat_pipeline_two = Pipeline(steps=[("label_encoder",FunctionTransformer(func=map_yes_no_to_binary))])
            cat_pipeline_three = Pipeline(steps=[("label_encoder",FunctionTransformer(func=map_gender))])

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_columns),
                ("cat_pipeline_one",cat_pipeline_one,categ_columns_one),
                ("cat_pipeline_two",cat_pipeline_two,categ_columns_two),
                ("cat_pipeline_three",cat_pipeline_three,categ_columns_three)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_excel(train_path)
            test_df=pd.read_excel(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_transformer_object()

            train_df = self.cleaner(train_df)
            test_df = self.cleaner(test_df)

            logging.info("Basic Cleaning done")

            target_column = 'Churn Value'

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info('Apply prepocessor on train and test data')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            
            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

