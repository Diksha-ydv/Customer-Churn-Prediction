from flask import Flask, render_template, request

from src.pipeline.prediction_pipeline import PredictData, CustomData

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "GET":
        return render_template("index.html")

    else:

        data = CustomData(

            customerID=request.form.get("customerID"),
            gender=request.form.get("gender"),
            SeniorCitizen=(request.form.get("SeniorCitizen")),
            Partner=request.form.get("Partner"),
            Dependents=request.form.get("Dependents"),
            tenure=int(request.form.get("tenure")),
            PhoneService=request.form.get("PhoneService"),
            MultipleLines=request.form.get("MultipleLines"),
            InternetService=request.form.get("InternetService"),
            OnlineSecurity=request.form.get("OnlineSecurity"),
            OnlineBackup=request.form.get("OnlineBackup"),
            DeviceProtection=request.form.get("DeviceProtection"),
            TechSupport=request.form.get("TechSupport"),
            StreamingTV=request.form.get("StreamingTV"),
            StreamingMovies=request.form.get("StreamingMovies"),
            Contract=request.form.get("Contract"),
            PaperlessBilling=request.form.get("PaperlessBilling"),
            PaymentMethod=request.form.get("PaymentMethod"),
            MonthlyCharges=float(request.form.get("MonthlyCharges")),
            TotalCharges=float(request.form.get("TotalCharges"))
        )

        pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictData()

        prediction = predict_pipeline.predict_data(pred_df)

        if prediction[0] == 1:
            result = "Customer is likely to Churn"
        else:
            result = "Customer is not likely to Churn"

        return render_template(
            "index.html",
            prediction=result
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)