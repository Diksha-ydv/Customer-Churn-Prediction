import os
import sys

import pandas as pd
import numpy as np

from scipy.stats import ks_2samp

from src.logger import logging
from src.exception import CustomerChurnException

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)

from src.entity.config_entity import DataValidationConfig
from src.constants.training_pipeline import SCHEMA_FILE_PATH

from src.utils import (
    read_data,
    save_yaml_file,
    read_yaml_file
)


class DataValidation:

    def __init__(self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def validate_no_of_columns(self, dataframe: pd.DataFrame):

        try:
            no_of_columns = len(self.schema_config["columns"])

            logging.info(f"Columns in schema : {no_of_columns}")
            logging.info(f"Columns in dataframe : {len(dataframe.columns)}")

            return no_of_columns == len(dataframe.columns)

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def validate_drift(self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05):

        try:

            report = {}
            status = True
            numerical_columns = self.schema_config["numerical_columns"]

            for column in numerical_columns:

                df1 = base_df[column]
                df2 = current_df[column]

                test = ks_2samp(df1, df2)

                if test.pvalue > threshold:
                    drift_detected = False
                else:
                    drift_detected = True
                    status = False

                report[column] = {
                    "p_value": float(test.pvalue),
                    "drift_detected": drift_detected
                }

            drift_report_file_path = (self.data_validation_config.drift_report_file_path)
            
            os.makedirs(
                os.path.dirname(drift_report_file_path),
                exist_ok=True
            )

            save_yaml_file(
                file_path=drift_report_file_path,
                data=report
            )

            return status

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def initiate_data_validation(self):

        try:

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Reading train and test data")

            train_df = read_data(train_file_path)
            test_df = read_data(test_file_path)

            logging.info("Validating number of columns")

            train_status = self.validate_no_of_columns(train_df)
            test_status = self.validate_no_of_columns(test_df)

            if not train_status or not test_status:
                raise Exception("Column validation failed.")

            logging.info("Column validation successful")

            logging.info("Performing data drift validation")

            drift_status = self.validate_drift(
                base_df=train_df,
                current_df=test_df
            )

            if drift_status:
                logging.info("No data drift detected.")
            else:
                logging.info("Data drift detected.")

            valid_train_file_path = (
                self.data_validation_config.valid_train_file_path
            )

            os.makedirs(
                os.path.dirname(valid_train_file_path),
                exist_ok=True
            )

            train_df.to_csv(
                valid_train_file_path,
                index=False,
                header=True
            )

            valid_test_file_path = (
                self.data_validation_config.valid_test_file_path
            )

            os.makedirs(
                os.path.dirname(valid_test_file_path),
                exist_ok=True
            )

            test_df.to_csv(
                valid_test_file_path,
                index=False,
                header=True
            )

            data_validation_artifact = DataValidationArtifact(

                validation_status=drift_status,

                valid_train_file_path=valid_train_file_path,

                valid_test_file_path=valid_test_file_path,

                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,

                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,

                drift_file_path=self.data_validation_config.drift_report_file_path

            )

            logging.info("Data validation completed successfully.")

            return data_validation_artifact

        except Exception as e:
            raise CustomerChurnException(e, sys)