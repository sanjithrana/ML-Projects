import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
from src.exception import CustomException
from src.logger import logging
from xgboost import XGBRegressor
import os
from src.utils import save_object,evaluate_model
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    trained_model_file= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting ,training,and test ")
            x_train,y_train,x_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "random forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K-Neightbors Reg": KNeighborsRegressor(),
                "XGBoosting Reg": XGBRegressor(),
                "CatBoost Reg": CatBoostRegressor(),
                "AdaBoost Reg": AdaBoostRegressor(),
            }
            params = {
                "random forest":{
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'n_estimators':[8,16,32,128,256]
                },
                "Decision Tree":{
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                },
                "Gradient Boosting":{
                    "loss": ['squared_error', 'absolute_error', 'huber', 'quantile'] ,
                    'criterion': ['friedman_mse', 'squared_error'],
                    "max_features": ['auto', 'sqrt', 'log2'],
                    'n_estimators':[8,16,32,128,256],
                    "learning_rate":[.1,.01,.05,.001],
                    "subsample":[0.6,0.7,0.8,0.05,0.9]

                },
                "K-Neightbors Reg":{
                    "n_neighbors": [5,7,8,9],
                    "weights": ['uniform', 'distance'],
                    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XGBoosting Reg":{
                    'n_estimators':[8,16,32,128,256],
                    "learning_rate":[.1,.01,.05,.001]
                },
                "CatBoost Reg":{
                    "depth":[6,8,10],
                    "learning_rate":[.1,.01,.05,.001],
                    "iterations":[30,50]

                },
                "AdaBoost Reg":{
                    "learning_rate":[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,128,256],
                    'loss':['linear', 'square', 'exponential'] 
                    

                }

            }
            # Evaluate all models
            model_report = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} | R2 Score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No model gave acceptable performance (R2 > 0.6).")
            
            best_model.fit(x_train, y_train)
            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file,
                obj=best_model
            )

            # Final prediction
            predictions = best_model.predict(x_test)
            final_score = r2_score(y_test, predictions)

            return final_score

        except Exception as e:
            raise CustomException(e, sys)

           