import os 
import sys 
import pandas as pd 
import yaml 
from src.exception import CustomerChurnException

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
            yaml.dump(data,file)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    
def read_yaml_file(file_path):
    '''This function is used to read a yaml file'''
    try:
        with open(file_path) as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    

