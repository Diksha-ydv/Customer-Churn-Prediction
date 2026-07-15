import os 
import sys 
import pandas as pd 
import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.logger import logging 
from src.exception import CustomerChurnException

from src.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig

from src.utils import read_data,save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_valiation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_valiation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def create_preprocessor(self):
        try:
            num_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "NewCustomer",
                "HighMonthlyCharges",
                "LongContract",
                "FiberOptic",
                "AutoPayment"
                ]
            
            cat_columns = [
                    "gender",
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "MultipleLines",
                    "InternetService",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "Contract",
                    "PaperlessBilling",
                    "PaymentMethod"
            ]

            num_pipeline = Pipeline(steps=[
                ("Imputer",SimpleImputer(strategy="median")),
                ("StandardScaler",StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("Imputer",SimpleImputer(strategy="most_frequent")),
                ("OneHotEncoder",OneHotEncoder(drop="if_binary",handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("Num_pipeline",num_pipeline,num_columns),
                ("Cat_pipeline",cat_pipeline,cat_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            train_data_path = self.data_validation_artifact.valid_train_file_path
            test_data_path = self.data_validation_artifact.valid_test_file_path

            logging.info("Reading validated data")

            train_data = read_data(file_path=train_data_path)
            test_data = read_data(file_path=test_data_path)

            # New Customer
            train_data["NewCustomer"] = (train_data["tenure"] <= 12).astype(int)
            test_data["NewCustomer"] = (test_data["tenure"] <= 12).astype(int)

            # Long Contract
            train_data["LongContract"] = (
                train_data["Contract"] != "Month-to-month"
            ).astype(int)

            test_data["LongContract"] = (
                test_data["Contract"] != "Month-to-month"
            ).astype(int)

            # Fiber Optic
            train_data["FiberOptic"] = (
                train_data["InternetService"] == "Fiber optic"
            ).astype(int)

            test_data["FiberOptic"] = (
                test_data["InternetService"] == "Fiber optic"
            ).astype(int)

            # Auto Payment
            train_data["AutoPayment"] = (
                train_data["PaymentMethod"].isin([
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ])
            ).astype(int)

            test_data["AutoPayment"] = (
                test_data["PaymentMethod"].isin([
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ])
            ).astype(int)

            # High Monthly Charges
            threshold = train_data["MonthlyCharges"].median()

            train_data["HighMonthlyCharges"] = (
                train_data["MonthlyCharges"] > threshold
            ).astype(int)

            test_data["HighMonthlyCharges"] = (
                test_data["MonthlyCharges"] > threshold
            ).astype(int)

            # Drop customerID
            train_data = train_data.drop(columns=["customerID"])
            test_data = test_data.drop(columns=["customerID"])

            # Replace blank values
            train_data["TotalCharges"] = train_data["TotalCharges"].replace(" ", np.nan)
            test_data["TotalCharges"] = test_data["TotalCharges"].replace(" ", np.nan)

            # Convert to float
            train_data["TotalCharges"] = pd.to_numeric(train_data["TotalCharges"], errors="coerce")
            test_data["TotalCharges"] = pd.to_numeric(test_data["TotalCharges"], errors="coerce")

            # separate X and y
            input_train_data = train_data.drop(columns=["Churn"])
            target_train_data = train_data["Churn"]

            input_test_data = test_data.drop(columns=["Churn"])
            target_test_data = test_data["Churn"]

            target_train_data = target_train_data.map({"No": 0, "Yes": 1})
            target_test_data = target_test_data.map({"No": 0, "Yes": 1})

            logging.info("Applying preprocessor to numerical and categorical columns")

            preprocessor = self.create_preprocessor()
            preprocessor_obj = preprocessor.fit(input_train_data)

            input_train_transformed_data = preprocessor_obj.transform(input_train_data)
            input_test_transformed_data = preprocessor_obj.transform(input_test_data)

            logging.info("Applying SMOTE on training data")

            smote = SMOTE(random_state=42)

            input_train_transformed_data, target_train_data = smote.fit_resample(
                input_train_transformed_data,
                    target_train_data)

            logging.info("SMOTE applied successfully")

            train_arr = np.c_[input_train_transformed_data,np.array(target_train_data)]
            test_arr = np.c_[input_test_transformed_data,np.array(target_test_data)]

            train_arr_path = self.data_transformation_config.transformed_train_file_path
            test_arr_path = self.data_transformation_config.transformed_test_file_path

            os.makedirs(os.path.dirname(train_arr_path),exist_ok=True)
            os.makedirs(os.path.dirname(test_arr_path),exist_ok=True)

            save_numpy_array_data(file_path=train_arr_path,data=train_arr)
            save_numpy_array_data(file_path=test_arr_path,data=test_arr)

            logging.info("Saved the transformed data")

            preprocessor_obj_path = self.data_transformation_config.transformed_obj_file_path

            os.makedirs(os.path.dirname(preprocessor_obj_path),exist_ok=True)
            save_object(file_path=preprocessor_obj_path,obj=preprocessor_obj)

            logging.info("Saving our preprocessor")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_obj_path=self.data_transformation_config.transformed_obj_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)






