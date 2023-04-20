from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from constants import TREE_FILE, LOGISTIC_FILE

def build_tree(X_train, y_train):
    clf_tree = DecisionTreeClassifier()
    clf_tree = clf_tree.fit(X_train, y_train)
    
    # Save model
    with open(TREE_FILE, 'wb') as file:
        pickle.dump(clf_tree, file)

    return clf_tree

def build_logistic(X_train, y_train):
    clf_logistic = LogisticRegression(max_iter=100)
    clf_logistic = clf_logistic.fit(X_train, y_train)
    
    # Save model
    with open(LOGISTIC_FILE, 'wb') as file:
        pickle.dump(clf_logistic, file)

    return clf_logistic