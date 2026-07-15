import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomerChurnException
from src.utils import load_object


class PredictData:

    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def predict_data(self, features):
        try:

            model = load_object("final model/model.pkl")
            preprocessor = load_object("final model/preprocessor.pkl")

            transformed_data = preprocessor.transform(features)

            # Probability of Churn = Yes
            probability = model.predict_proba(transformed_data)[:, 1]

            # Use the threshold obtained during training
            threshold = 0.41 

            prediction = (probability >= threshold).astype(int)

            return prediction

        except Exception as e:
            raise CustomerChurnException(e, sys)


class CustomData:

    def __init__(
        self,
        customerID,
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges
    ):

        self.customerID = customerID
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_dataframe(self):

        try:

            df = pd.DataFrame({

                "customerID": [self.customerID],
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]

            })

            # ---------------- Feature Engineering ---------------- #

            df["NewCustomer"] = (df["tenure"] <= 12).astype(int)

            df["LongContract"] = (
                df["Contract"].isin(["One year", "Two year"])
            ).astype(int)

            df["FiberOptic"] = (
                df["InternetService"] == "Fiber optic"
            ).astype(int)

            df["AutoPayment"] = (
                df["PaymentMethod"].isin(
                    [
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )
            ).astype(int)

            df["HighMonthlyCharges"] = (
                df["MonthlyCharges"] >= 70
            ).astype(int)

            logging.info("Prediction dataframe created successfully.")

            return df

        except Exception as e:
            raise CustomerChurnException(e, sys)