import os 
import sys 
import pandas as pd 

from src.logger import logging
from src.exception import CustomerChurnException

from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from src.utils import (load_numpy_array_data,evaluate_model, load_object, save_object,
                       save_yaml_file,classification_metric,tune_threshold)

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def train_data(self,x_train,y_train,x_test,y_test):
        try:
            models = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                random_state=42,
                verbose=0
            ),

            "Decision Tree Classifier": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),

            "Random Forest Classifier": RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                verbose=1
            ),

            "Gradient Boost Classifier": GradientBoostingClassifier(
                random_state=42,
                verbose=1
            ),

            "Adaboost classifier": AdaBoostClassifier(
                random_state=42
            ),

            "K Neighbors Classifier": KNeighborsClassifier(),

            "XG boost classifier": XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            ),

            "CatBoost Clasifier": CatBoostClassifier(
                random_state=42,
                verbose=0
            )}

            complete_report, test_report = evaluate_model(
                        x_train=x_train,
                        x_test=x_test,
                        y_train=y_train,
                        y_test=y_test,
                        models=models
                    )

            # Saving the complete report
            complete_report_path = self.model_trainer_config.model_trainer_report_dir
            os.makedirs(os.path.dirname(complete_report_path), exist_ok=True)
            save_yaml_file(file_path=complete_report_path, data=complete_report)

            # Find the best model
            test_report_df = pd.DataFrame(
                test_report.items(),
                columns=["Model Name", "Score"]
            ).sort_values(by="Score", ascending=False)

            best_model_name = test_report_df.iloc[0, 0]
            best_model_score = test_report_df.iloc[0, 1]

            logging.info(f"Best model is {best_model_name} with score {best_model_score}")

            # Get the already-trained best model
            best_model = models[best_model_name]

            # Threshold tuning
            best_threshold, threshold_f1 = tune_threshold(
                model=best_model,
                x_test=x_test,
                y_test=y_test
            )

            logging.info(f"Best Threshold : {best_threshold}")
            logging.info(f"Best F1 Score after threshold tuning : {threshold_f1}")

            # Predictions using best threshold
            y_train_probability = best_model.predict_proba(x_train)[:, 1]
            y_train_pred = (y_train_probability >= best_threshold).astype(int)

            y_test_probability = best_model.predict_proba(x_test)[:, 1]
            y_test_pred = (y_test_probability >= best_threshold).astype(int)

            classification_train_metric = classification_metric(
                y_true=y_train,
                y_pred=y_train_pred
            )

            classification_test_metric = classification_metric(
                y_true=y_test,
                y_pred=y_test_pred
            )

            # Save best model report
            best_model_report = {
                "Best Model": best_model_name,
                "Best Threshold": float(best_threshold),
                "Threshold F1 Score": float(threshold_f1),

                "Train Metrics": {
                    "Accuracy": classification_train_metric.accuracy_score,
                    "Precision": classification_train_metric.precision_score,
                    "Recall": classification_train_metric.recall_score,
                    "F1 Score": classification_train_metric.f1_score
                },

                "Test Metrics": {
                    "Accuracy": classification_test_metric.accuracy_score,
                    "Precision": classification_test_metric.precision_score,
                    "Recall": classification_test_metric.recall_score,
                    "F1 Score": classification_test_metric.f1_score
                }
            }

            best_model_path = self.model_trainer_config.best_model_report_path
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            save_yaml_file(file_path=best_model_path, data=best_model_report)

            logging.info("Saving the best model")

            model_path = self.model_trainer_config.trained_model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            save_object(file_path=model_path, obj=best_model)

            preprocessor_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_obj_path
            )

            # Save final model
            os.makedirs("final model", exist_ok=True)

            save_object(
                file_path="final model/preprocessor.pkl",
                obj=preprocessor_obj
            )

            save_object(
                file_path="final model/model.pkl",
                obj=best_model
            )

            return classification_train_metric, classification_test_metric
        except Exception as e:
            raise CustomerChurnException(e,sys)
                    
    def initiate_model_trainer(self):
        try:
            train_arr_path = self.data_transformation_artifact.transformed_train_file_path
            test_arr_path = self.data_transformation_artifact.transformed_test_file_path

            #reading the array data 
            train_arr = load_numpy_array_data(train_arr_path)
            test_arr = load_numpy_array_data(test_arr_path)

            x_train,y_train,x_test,y_test = [
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            ]

            classification_train_metric,classification_test_metric = self.train_data(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model,
                train_score_metric=classification_train_metric,
                test_score_metric=classification_test_metric
            )
            return model_trainer_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)
