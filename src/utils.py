import os 
import sys 
import pandas as pd 
import numpy as np 
import yaml 
from src.exception import CustomerChurnException
import pickle 
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from src.entity.artifact_entity import ClassificationMetric

def read_data(file_path):
    '''This function is used to read a csv file'''
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    

def save_yaml_file(file_path,data):
    '''This function is used to save data into a yaml file'''
    try:
        with open(file_path,"w") as file:
            yaml.safe_dump(data,file)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    
    
def read_yaml_file(file_path):
    '''This function is used to read a yaml file'''
    try:
        with open(file_path) as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    
def save_numpy_array_data(file_path,data):
    '''This function is used to save array data '''
    try:
        with open(file_path,"wb") as file:
            np.save(file,data)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    

def save_object(file_path,obj):
    '''This function is used to save an object'''
    try:
        with open(file_path,"wb") as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    

def load_numpy_array_data(file_path):
    '''This function is used to load array data'''
    try:
        with open(file_path,"rb") as file:
            return np.load(file) 
    except Exception as e:
        raise CustomerChurnException(e,sys)
    
    
def evaluate_model(x_train,x_test,y_train,y_test,models):
    '''This function is used to evaluate the performance of our models'''
    try:
        complete_report = {}
        test_report = {}
        for name,model in models.items():
            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            y_test_pred_prob = model.predict_proba(x_test)[:,1]
            y_train_pred_prob = model.predict_proba(x_train)[:,1]

            f1_train = f1_score(y_train,y_train_pred)
            precision_train = precision_score(y_train,y_train_pred,zero_division=0)
            recall_train = recall_score(y_train,y_train_pred,zero_division=0)
            accuracy_train = accuracy_score(y_train,y_train_pred)
            roc_auc_train = roc_auc_score(y_train,y_train_pred_prob)

            f1_test = f1_score(y_test,y_test_pred)
            precision_test = precision_score(y_test,y_test_pred,zero_division=0)
            recall_test = recall_score(y_test,y_test_pred,zero_division=0)
            accuracy_test = accuracy_score(y_test,y_test_pred)
            roc_auc_test = roc_auc_score(y_test,y_test_pred_prob)

            complete_report.update({
                name:{"train set result":
                    {"f1 score":f1_train,"precision score":precision_train,"recall score":recall_train,
                    "accuracy score":accuracy_train,"ROC AUC Score":roc_auc_train},

                    "test set result":{
                        "f1 score":f1_test,"precision score":precision_test,"recall score":recall_test,
                        "accuracy score":accuracy_test,"ROC AUC Score":roc_auc_test
                    }}
            })

            test_report[name] = f1_test
        return complete_report,test_report
    except Exception as e:
        raise CustomerChurnException(e,sys)
    

def load_object(file_path):
    '''This function is used to load an object'''
    try:
        with open(file_path,"rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    

def classification_metric(y_true,y_pred):
    '''This function is used to get the classification metric'''
    try:
        f1 = f1_score(y_true,y_pred)
        accuracy = accuracy_score(y_true,y_pred)
        precision = precision_score(y_true,y_pred,zero_division=0)
        recall = recall_score(y_true,y_pred,zero_division=0)

        metric = ClassificationMetric(f1_score=f1,precision_score=precision,accuracy_score=accuracy,recall_score=recall)
        return metric 
        
    except Exception as e:
        raise CustomerChurnException(e,sys)


def tune_threshold(model, x_test, y_test):
    
    '''This function is used to find the threshold giving maximum F1-score.'''
    try:
        probabilities = model.predict_proba(x_test)[:,1]

        best_threshold = 0.5
        best_score = 0

        for threshold in np.arange(0.10,0.91,0.01):

            prediction = (probabilities >= threshold).astype(int)

            score = f1_score(y_test,prediction)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold,best_score
    except Exception as e:
        raise CustomerChurnException(e,sys)




        

    





        
    

