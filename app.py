from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap
import os

from io import BytesIO
#import matplotlib.pyplot as plt
from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Hello, this is the home page!'

MODEL_PATH = 'loan_model.pkl'
DATA_PATH = 'client_data.csv'
RAW_DATA_PATH = 'Xtest_raw.csv'
THRESHOLD_PATH = 'threshold.txt'

model = joblib.load(MODEL_PATH)
df_test = pd.read_csv(DATA_PATH)
df_test_raw = pd.read_csv(RAW_DATA_PATH)
with open(THRESHOLD_PATH, 'r') as file:
    custom_threshold = float(file.read())

explainer = shap.TreeExplainer(model)
df_test_exp = df_test.drop("SK_ID_CURR", axis=1)
shap_values = explainer.shap_values(df_test_exp)

@app.route('/predict/', methods=['POST'])
def predict():
    data = request.json
    client_code_1 = int(data["client_code_1"])
    selected_client_data_1 = df_test[df_test['SK_ID_CURR'] == client_code_1].drop(['SK_ID_CURR'], axis=1)
    prob_1 = model.predict_proba(selected_client_data_1)[:, 1][0]

    predictions = {
        "prediction_1": "Granted" if prob_1 < custom_threshold else "Not Granted",
        "probability_1": prob_1 * 100
      
    }
    
    idx = df_test_raw[df_test_raw['SK_ID_CURR'] == client_code_1].index[0]
    exp = shap.Explanation(values=shap_values[1][idx, :],
                           base_values=explainer.expected_value[1],
                           data=df_test_exp.iloc[idx, :])

    predictions['values']= exp.values.tolist()
    predictions['base_values']= exp.base_values
    
    dat = exp.data.to_dict()

    combined_data = {
    "predictions": predictions,
    "dat": dat
}

    return jsonify(combined_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
