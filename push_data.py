import pymongo 
import os 
import sys 
import pandas as pd 
import json 

from src.exception import CustomerChurnException
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class ExportData:
    def __init__(self):
        try:
            pass 
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def data_json_convert(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)

            records = list(json.loads(data.T.to_json()).values())
            return records 
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def push_data_mongodb(self,records,database,collection):
        try:
            self.records = records 
            self.database = database 
            self.collection = collection 

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]

            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise CustomerChurnException(e,sys)
        

if __name__=="__main__":
    export_data = ExportData()
    file_path = "Customer_Data/customer_churn_data.csv"
    database = "Diksha"
    collection = "Customer_Churn_Data"
    records = export_data.data_json_convert(file_path=file_path)
    no_of_records = export_data.push_data_mongodb(records=records,database=database,collection=collection)
    print(f"Number of records = {no_of_records}")
