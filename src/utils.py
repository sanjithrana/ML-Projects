#common fun that we can use
import os
import sys
import pickle
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score

import numpy as np  
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
import sys
from sklearn.model_selection import GridSearchCV
def evaluate_model(x_train, y_train, x_test, y_test, models: dict, params: dict):
    try:
        report = {}

        for name, model in models.items():
            param = params[name]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_

            y_pred = best_model.predict(x_test)
            score = r2_score(y_test, y_pred)

            report[name] = score   # ONLY FLOAT SCORE

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
       with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
            raise CustomException(e, sys)
    