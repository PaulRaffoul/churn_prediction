import os
import sys

import numpy as np 
import pandas as pd
#import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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
    print(type(X))
    X.map({"Yes": 1, "No": 0})
    return X
    
def map_gender(X):
    return X.map({"Female": 1, "Male": 0})