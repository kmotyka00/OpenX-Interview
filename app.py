from flask import Flask, request, jsonify
import data
import heuristics
from models import build_logistic, build_tree  # build nn
from constants import DATASET_URL, HEURISTIC_FILE, TREE_FILE, LOGISTIC_FILE, NN_FILE
import os
import pandas as pd
import pickle 

# Create app
app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    df = data.load_data(url=DATASET_URL,
                        column_names=data.get_column_names())
    normalized_df, scaler = data.preprocess_data(df)
    X_train, X_test, y_train, y_test = data.split_data(normalized_df)
    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
# Load or create models
try:
    heuristic = pd.read_pickle(HEURISTIC_FILE) #TODO
except:
    heuristic = heuristics.create_heuristic(X_train, y_train)

try:
    with open(TREE_FILE, 'rb') as file:
        clf_tree = pickle.load(file)
except:      
    clf_tree = build_tree(X_train, y_train)

try:
    with open(LOGISTIC_FILE, 'rb') as file:
        clf_log = pickle.load(file)
except:     
    clf_log = build_logistic(X_train, y_train)

# Create routes
@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    input_data = req["data"]
    model_type = req["model"]

    # Normalize input data
    input_data, _ = data.normalize_input(input_data, scaler)

    # Predict with selected model
    if model_type == "heuristic":
        preds = heuristics.predict(input_data, heuristic)
    elif model_type == "decision_tree":
        preds = clf_tree.predict(input_data)

    elif model_type == "logreg":
        preds = clf_log.predict(input_data)
    elif model_type == "neural_network":
        pass
    else:
        return jsonify({'error': 'Invalid model selection'})
    
    if type(preds) is not list:
        preds = preds.tolist()
    return jsonify({"prediction": preds})


if __name__ == '__main__':
    app.run(debug=True)
