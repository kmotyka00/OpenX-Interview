import os

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'

MODELS_DIRECTORY = r"Models"

HEURISTIC_FILE = os.path.join(MODELS_DIRECTORY, "heuristic.pkl")
TREE_FILE = os.path.join(MODELS_DIRECTORY, "decision_tree_model.pkl")
LOGISTIC_FILE = os.path.join(MODELS_DIRECTORY, "logistic_model.pkl")
NN_FILE = os.path.join(MODELS_DIRECTORY, "neural_network_model.h5")
GRID_SEARCH_FILE = os.path.join(MODELS_DIRECTORY, "grid_search.pkl")
NN_HISTORY_FILE = os.path.join(MODELS_DIRECTORY, "history.pkl")