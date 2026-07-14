import os 

'''Defining some constants'''
TARGET_COLUMN = "Churn"
DATA_FILE_NAME = "customer_churn_data.csv"
ARTIFACTS_DIR_NAME = "Artifacts"
TRAIN_FILE_NAME = "Train.csv"
TEST_FILE_NAME = "Test.csv"

'''Defining some constants for data ingestion'''
DATA_INGESTION_DIR = "Data Ingestion"
DATA_INGESTION_INGESTED_DIR = "Ingested"
DATA_INGESTION_FEATURE_DIR = "Features"
DATA_INGESTION_TRAIN_TEST_RATIO = 0.2 
DATA_INGESTION_DATABASE_NAME = "Diksha"
DATA_INGESTION_COLLECTION_NAME = "Customer_Churn_Data"

'''Defining some constants for data validation'''
DATA_VALIDATION_DIR = "Data Validation"
DATA_VALIATION_VALID_DIR = "Valid"
DATA_VALIDATION_INVALID_DIR = "Invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR = "Drift Report"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME = "report.yaml"

SCHEMA_FILE_PATH = os.path.join("Data Schema","schema.yaml")


'''Defining some constants for data transformation'''


'''Defining some constants for model training'''
