import pickle 
from machine_learning.sklearn_models import build_logistic, build_tree
import tensorflow as tf
from config_files.constants import TREE_FILE, LOGISTIC_FILE, NN_FILE, DATASET_URL
from machine_learning.sklearn_models import build_logistic, build_tree
from machine_learning.neural_network import build_nn
import pickle
import machine_learning.heuristics as heuristics
import data_preprocessing.data as data


def load_and_preprocess_data():
    df = data.load_data(url=DATASET_URL,
                        column_names=data.get_column_names())
    normalized_df, scaler = data.preprocess_data(df)
    X_train, X_test, y_train, y_test = data.split_data(normalized_df)
    return X_train, X_test, y_train, y_test, scaler

def load_or_create_models(X_train, y_train):

    heuristic = heuristics.Heuristic(X_train, y_train)
    
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

    try:
        clf_nn = tf.keras.models.load_model(NN_FILE)
    except:
        clf_nn = build_nn(X_train, y_train)

    return heuristic, clf_tree, clf_log, clf_nn