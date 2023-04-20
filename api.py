from flask import Flask, request, jsonify
import data
from app_utils import load_and_preprocess_data, load_or_create_models
import numpy as np

# Create app
app = Flask(__name__)

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

# Load or create models
heuristic, clf_tree, clf_log, clf_nn = load_or_create_models(X_train, y_train)

# Create routes
@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    # TODO: try except
    input_data = req["data"]
    model_type = req["model"]

    # Normalize input data
    input_data, _ = data.normalize_input(input_data, scaler)

    # Predict with selected model
    if model_type == "heuristic":
        preds = heuristic.predict(input_data)

    elif model_type == "decision_tree":
        preds = clf_tree.predict(input_data)

    elif model_type == "logistic_regression":
        preds = clf_log.predict(input_data)

    elif model_type == "neural_network":
        mapping = np.unique(y_train).tolist()
        label = np.vectorize(lambda x: mapping[x])
        preds = label(np.argmax(clf_nn.predict(input_data), axis=1))
    else:
        return jsonify({'error': 'Invalid model selection'})

    if type(preds) is not list:
        preds = preds.tolist()

    # Return prediction
    return jsonify({"prediction": preds,
                    "model": model_type})


# Run app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
