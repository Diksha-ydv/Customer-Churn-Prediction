import os 
from datetime import datetime 
from src.constants import training_pipeline


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        self.timestamp = timestamp
        self.artifact_name = training_pipeline.ARTIFACTS_DIR_NAME
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)


class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config

        self.data_ingestion_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR
        )
        self.data_ingestion_ingested_dir = os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR)
        self.train_file_path = os.path.join(self.data_ingestion_ingested_dir,training_pipeline.TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(
            self.data_ingestion_ingested_dir,training_pipeline.TEST_FILE_NAME
        )

        self.data_ingestion_feature_store_path = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_DIR,training_pipeline.DATA_FILE_NAME)
        self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        self.data_ingestion_train_test_split_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_RATIO


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config

        self.data_validation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,training_pipeline.DATA_VALIDATION_DIR)
        self.data_validation_valid_dir = os.path.join(
            self.data_validation_dir,training_pipeline.DATA_VALIATION_VALID_DIR
        )
        self.valid_train_file_path = os.path.join(self.data_validation_valid_dir,training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path = os.path.join(self.data_validation_valid_dir,training_pipeline.TEST_FILE_NAME)

        self.data_validation_invalid_dir = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.invalid_train_file_path = os.path.join(self.data_validation_invalid_dir,training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(self.data_validation_invalid_dir,training_pipeline.TEST_FILE_NAME)

        self.drift_report_file_path = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                   training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME)
        
        
        