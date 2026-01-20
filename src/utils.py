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
    
def evaluate_model(x_train,y_train,y_test,x_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            models = list(models.values())[i]

            models.fit(x_train,y_train)

            y_train_pred = models.predict(x_train)
            y_test_pred = models.predict(y_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = train_model_score
        return report
    except Exception as e:
        return CustomException(e,sys)
    