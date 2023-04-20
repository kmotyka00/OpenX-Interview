import pandas as pd 
import numpy as np
from constants import HEURISTIC_FILE

def create_heuristic(X_train, y_train):
    train = pd.concat([X_train, y_train], axis=1)
    avg_cover_type = train.groupby('Cover_Type').mean()

    # Save the heuristic
    avg_cover_type.to_pickle(HEURISTIC_FILE)
    
    return avg_cover_type

def predict(test_data, heuristic):
    
    # Get cover type labels
    mapping = heuristic.index.values
    label = np.vectorize(lambda x: mapping[x])
    
    # Calculate the difference between the test data and the heuristic
    diff_df = np.abs(heuristic.values - test_data.to_numpy()[:, None, :])

    # Sum the differences for each cover type
    ranking_df = diff_df.sum(axis=2)

    # Get the cover type with the lowest difference
    preds = np.argmin(ranking_df, axis=1)

    return label(preds)