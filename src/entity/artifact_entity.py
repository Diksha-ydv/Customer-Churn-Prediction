from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str 

@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_train_file_path:str 
    valid_test_file_path:str 
    invalid_train_file_path:str 
    invalid_test_file_path:str 
    drift_file_path:str 

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str 
    transformed_test_file_path:str 
    transformed_obj_path : str 
    

@dataclass
class ClassificationMetric:
    f1_score:float 
    precision_score:float 
    recall_score:float 
    accuracy_score:float


@dataclass 
class ModelTrainerArtifact:
    trained_model_path:str 
    test_score_metric:ClassificationMetric
    train_score_metric:ClassificationMetric