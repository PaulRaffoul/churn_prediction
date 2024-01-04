import os
import sys
import json

import numpy as np 
import pandas as pd
#import dill
import pickle
from sklearn.metrics import recall_score,accuracy_score,f1_score,precision_recall_curve
from sklearn.model_selection import cross_val_predict,GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def map_yes_no_to_binary(X):
    # get columns of X
    cols = X.columns
    #iterate
    for col in cols:
        X[col] = X[col].map({"Yes": 1, "No": 0})
    return X
    
def map_gender(X):
    # get columns of X
    cols = X.columns
    #iterate
    for col in cols:
        X[col] = X[col].map({"Female": 1, "Male": 0})
    return X

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = recall_score(y_train, y_train_pred)

            test_model_score = recall_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def save_json(file_path,value):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump({"threshold": value}, f)

    except Exception as e:
        raise CustomException(e, sys)

    
def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e, sys)
        
    
def find_threshold(model,y_train,X_train,target_recall=0.9):
    y_scores = cross_val_predict(model,X_train,y_train,cv=3,method='decision_function')
    precisons,recalls,thresholds = precision_recall_curve(y_train,y_scores)
    differences = np.abs(recalls - target_recall)
    closest_index = np.argmin(differences)
    threshold = thresholds[closest_index]
    return threshold