import os 
import sys 
import pandas as pd 
import pymongo 
import numpy as np 

from src.logger import logging 
from src.exception import CustomerChurnException

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config


    def import_data(self):
        try:
            self.database = self.data_ingestion_config.database_name
            self.collection = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[self.database][self.collection]

            logging.info("Importing the data from the database")
            records = pd.DataFrame(list(collection.find()))
            if "_id" in records.columns:
                records.drop(columns="_id",inplace=True)

            records.replace({"na":np.nan},inplace=True)
            return records 
        except Exception as e:
            raise CustomerChurnException(e,sys)
    
    def initiate_data_ingestion(self):
        try:
            records = self.import_data()
            feature_store_path = self.data_ingestion_config.data_ingestion_feature_store_path

            os.makedirs(os.path.dirname(feature_store_path),exist_ok=True)
            records.to_csv(feature_store_path,index=None,header=True)
            
            y = records["Churn"]
            train_data,test_data = train_test_split(
                records,
                test_size=self.data_ingestion_config.data_ingestion_train_test_split_ratio,
                random_state=42,
                stratify=y)
            
            logging.info("Train test split done")

            train_file_path = self.data_ingestion_config.train_file_path
            os.makedirs(os.path.dirname(train_file_path),exist_ok=True)
            train_data.to_csv(train_file_path,index=None,header=True)

            test_file_path = self.data_ingestion_config.test_file_path
            os.makedirs(os.path.dirname(test_file_path),exist_ok=True)
            test_data.to_csv(test_file_path,index=None,header=True)

            logging.info("Saved the data into train and test file")

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)

