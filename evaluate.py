from sklearn import metrics

def accuracy(y_true, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_true, y_pred))

