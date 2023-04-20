import sys
sys.path.append(".")

from sklearn import metrics
from rest_api.app_utils import load_and_preprocess_data, load_or_create_models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from config_files.constants import NN_HISTORY_FILE

def accuracy(y_true, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_true, y_pred))


def print_confusion_matrix(y_true, y_pred):
    print("Confusion Matrix:\n", metrics.confusion_matrix(y_true, y_pred))


def plot_confusion_matrix(model_name, y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion matrix for {model_name}")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def classification_report(y_true, y_pred):
    print(metrics.classification_report(y_true, y_pred))


def evaluate(y_true, y_pred):
    accuracy(y_true, y_pred)
    classification_report(y_true, y_pred)
    print_confusion_matrix(y_true, y_pred)
    print("\n\n")

def load_nn_history(file_path):
    try:
        history = joblib.load(file_path)
    except:
        print("No history found")
        return None
    return history

def plot_learning_curve(history):

    fig, axs = plt.subplots(2, 1, figsize=(5,10))

    plt.suptitle("Learning curves")
    axs[0].plot(history.history['accuracy'])
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')

    axs[1].plot(history.history['loss'])
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    
    plt.show()

if __name__ == "__main__":

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    mapping = np.unique(y_train).tolist()
    label = np.vectorize(lambda x: mapping[x])
   
    # Load or create models
    heuristic, clf_tree, clf_log, clf_nn = load_or_create_models(
        X_train, y_train)

    # Create dictionary of models
    models = {
        "Heuristic": heuristic,
        "Decision Tree": clf_tree,
        "Logistic Regression": clf_log,
        "Neural Network": clf_nn
    }

    results = dict()

    # Plot learning curve
    history = load_nn_history(NN_HISTORY_FILE)
    if history:
        plot_learning_curve(history)

    # Make predictions with each model
    for model_name, model in models.items():

        if model_name == "Neural Network":
            predictions = label(
                np.argmax(clf_nn.predict(X_test, verbose=0), axis=1))
        else:
            predictions = model.predict(X_test)

        results[model_name] = predictions

    # Evaluate each model
    for model_name, predictions in results.items():
        print("## ", model_name.upper(), " ##")
        evaluate(y_test, predictions)
        plot_confusion_matrix(model_name, y_test, predictions)
