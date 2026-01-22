# this the a place where data is present
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd  
import os
import sys
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainingConfig

@dataclass #because inside a class to define the class variable we use __init__
           #but with the help of this we can directly define the class variable
class DataIngenstionConfig:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")

#here we need the functions so we are not using the decarator

class DataIngenstion:
    def __init__(self):
        self.ingenstion_config = DataIngenstionConfig()
    def initiate_ingenstion_data(self):#we our data is stored in any database we have to call
        logging.info("Enter the data ingestion method or components")
        try:
            df= pd.read_csv("src//notebook//data//stud.csv")
            logging.info('Read the Dataest as DataFrame')

            os.makedirs(os.path.dirname(self.ingenstion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingenstion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initited")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingenstion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingenstion_config.test_data_path,index = False,header = True)

            logging.info("ingenstion of the data is happen")

            return( self.ingenstion_config.train_data_path,
                     self.ingenstion_config.test_data_path
                     )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":#initiate
    obj = DataIngenstion()
    train_data,test_data = obj.initiate_ingenstion_data()

    data_transformation = DataTransformation()
    train_arr,test_arr ,_= data_transformation.initiate_data_transformation(train_data,test_data)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))