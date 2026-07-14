import os 
import sys

from src.logger import logging 
from src.exception import CustomerChurnException

from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from src.components.data_ingestion import DataIngestion

class TrainingPipeline:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config
        except Exception as e:
            raise CustomerChurnException(e,sys)

    def start_data_ingestion(self):
        try:
            logging.info("Data ingestion started")
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifacts = {data_ingestion_artifacts}")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomerChurnException(e,sys)

    def running_pipeline(self):
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
        except Exception as e:
            raise CustomerChurnException(e,sys)