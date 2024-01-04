import os
import sys
import json

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score,f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models,find_threshold,load_json,save_json

class ModelTrainer:
    def __init__(self):
        self.trained_model_file_path = os.path.join("artifacts","model.pkl")
        self.threshold_path = os.path.join("artifacts","threshold.json")
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
                # "K Neighbors":KNeighborsClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Logistic Regression":LogisticRegression(max_iter=200),
                "Support Vector":SVC(),
            }
            params={
                "Decision Tree": {
                    'criterion':['entropy','gini'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "K Neighbors":{
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Logistic Regression":{
                    'C': [0.1, 1, 10],
                },
                "Support Vector":{
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },  
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(best_model)

            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.trained_model_file_path,
                obj=best_model
            )

            #get threshold for recall 90 percent
            threshold = find_threshold(model=best_model,y_train=y_train,X_train=X_train,target_recall=0.9)
            save_json(self.threshold_path,value=threshold)

            logging.info(f"Threshold for high recall generated and saved")

            print(threshold)

            # predicted=best_model.predict(X_test)

            # score = recall_score(y_test, predicted)
            # return score
            
        except Exception as e:
            raise CustomException(e,sys)
            

            

