from sklearn import metrics
import heuristics
import data
import pandas as pd
from constants import DATASET_URL, HEURISTIC_FILE, TREE_FILE, LOGISTIC_FILE, NN_FILE
from models import build_logistic, build_tree
from neural_network import build_nn
import pickle
import tensorflow as tf
from app_utils import load_and_preprocess_data, load_or_create_models
import numpy as np

def accuracy(y_true, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_true, y_pred))

def confusion_matrix(y_true, y_pred):
    print("Confusion matrix:\n", metrics.confusion_matrix(y_true, y_pred))

def classification_report(y_true, y_pred):
    print("Classification report:\n", metrics.classification_report(y_true, y_pred))

def evaluate(y_true, y_pred):   
    accuracy(y_true, y_pred)
    confusion_matrix(y_true, y_pred)
    classification_report(y_true, y_pred)

def evaluate_models(models, X_test, y_test):
    for model in models:
        print(model)
        y_pred = model.predict(X_test)
        evaluate(y_test, y_pred)

def load_and_preprocess_data():
    df = data.load_data(url=DATASET_URL,
                        column_names=data.get_column_names())
    normalized_df, scaler = data.preprocess_data(df)
    X_train, X_test, y_train, y_test = data.split_data(normalized_df)
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    mapping = np.unique(y_train).tolist()
    label = np.vectorize(lambda x: mapping[x])
    # Load or create models
    heuristic, clf_tree, clf_log, clf_nn = load_or_create_models(X_train, y_train)

    models = {
        "Heuristic": heuristic,
        "Decision Tree": clf_tree,
        "Logistic Regression": clf_log,
        "Neural Network": clf_nn
    }

    results = dict()

    for model_name, model in models.items():
 
        if model_name == "Neural Network":
            predictions = label(np.argmax(clf_nn.predict(X_test, verbose=0), axis=1))
        else:
            predictions = model.predict(X_test)

        results[model_name] = predictions
    
    for model_name, predictions in results.items():
        print(model_name, end=" ")
        accuracy(y_test, predictions)



    
    