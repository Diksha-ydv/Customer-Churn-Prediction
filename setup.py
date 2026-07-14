from setuptools import setup,find_packages
from typing import List 


def get_requirements(file_path)->List[str]:
    '''This function is used to get list of requirements'''
    Hyphen = "-e ."
    with open(file_path) as file:
        requirement_lst = file.readlines()
        requirement_lst = [line.strip() for line in requirement_lst]
    
    if Hyphen in requirement_lst:
        requirement_lst.remove(Hyphen)

    print(requirement_lst)
    return requirement_lst

setup(
    name="Customer-Churn-Prediction",
    version="0.0.1",
    author="Diksha",
    author_email="dikshaydv2006@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)