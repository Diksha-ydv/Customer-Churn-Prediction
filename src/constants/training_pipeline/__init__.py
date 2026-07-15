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
DATA_TRANSFORMATION_DIR = "Data Transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR = "Transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJ_DIR = "Transformed_Obj"
TRANSFORMED_OBJ_NAME = "preprocessor.pkl"


'''Defining some constants for model training'''
MODEL_TRAINER_DIR = "Model Trainer"
MODEL_TRAINER_TRAINED_DIR = "Trained"
TRAINED_MODEL_NAME = "model.pkl"
MODEL_TRAINED_REPORT = "model_report.yaml"
MODEL_TRAINER_REPORT_DIR = "Report"
MODEL_TRAINER_BEST_MODEL_REPORT = "best_model.yaml"
