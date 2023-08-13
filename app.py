from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

def load_model():
    with open('results/model.pckl', 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess(df):
    with open("results/encoder.pckle", "rb") as f:
        encoder = pickle.load(f)
    with open("results/imputer.pckle", "rb") as f:
        imputer = pickle.load(f)
    df["smoking_status"].replace("Unknown", np.nan, inplace=True)
    df = encoder.transform(df.loc[:,["gender", "hypertension", "heart_disease",
                                "ever_married", "work_type", "Residence_type", "smoking_status",
                                "stroke"]])
    data_frame = imputer.transform(df.drop("stroke", axis=1))
    return data_frame
    

app = Flask(__name__)
df = pd.DataFrame([["Male", 22, 0, 0, "No", "Private", "Rural", 250, 32, "smokes", 0]], 
                columns=["gender","age","hypertension","heart_disease","ever_married"
                ,"work_type","Residence_type","avg_glucose_level","bmi","smoking_status","stroke"])


@app.route('/predict', methods=['POST'])
def predict():
    features = preprocess(request.json['features'])
    model = load_model()
    prediction = model.predict(features)
    return jsonify({"Prediction":prediction})
if __name__ == "__main__":
    app.run(debug=True)