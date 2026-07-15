import sys

from src.components.model_trainer import ModelTrainer
from src.logger import logging 
from src.exception import CustomerChurnException

from src.entity.config_entity import (TrainingPipelineConfig,DataIngestionConfig,
                                      DataValidationConfig,DataTransformationConfig,
                                      ModelTrainerConfig)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

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
        
    def start_data_validation(self,data_ingestion_artifacts):
        try:
            logging.info("Data validation started")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifacts)
            data_validation_artifacts = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed and artifacts = {data_validation_artifacts}")
            return data_validation_artifacts
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def start_data_transformation(self,data_validation_artifacts):
        try:
            logging.info("Data transformation started")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_transformation_config=data_transformation_config,data_valiation_artifact=data_validation_artifacts)
            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed and artifacts = {data_transformation_artifacts}")
            return data_transformation_artifacts
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def start_model_training(self,data_transformation_artifacts):
        try:
            logging.info("Model training started")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifacts)
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info("Model training completed")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomerChurnException(e,sys)

    def running_pipeline(self):
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_validation_artifacts = self.start_data_validation(data_ingestion_artifacts=data_ingestion_artifacts)
            data_transformation_artifacts = self.start_data_transformation(data_validation_artifacts=data_validation_artifacts)
            model_trainer_artifacts = self.start_model_training(data_transformation_artifacts=data_transformation_artifacts)
            logging.info("Training pipeline completed successfully")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomerChurnException(e,sys)