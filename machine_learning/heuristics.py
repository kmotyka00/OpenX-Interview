import pandas as pd 
import numpy as np
from config_files.constants import HEURISTIC_FILE
import os 

class Heuristic:
    def __init__(self, X_train, y_train, heuristic_file=HEURISTIC_FILE):
        if os.path.exists(heuristic_file):
            self.metric = pd.read_pickle(heuristic_file)
        else:
            self.create_heuristic(X_train, y_train)

    def create_heuristic(self, X_train, y_train):
        train = pd.concat([X_train, y_train], axis=1)
        avg_cover_type = train.groupby('Cover_Type').mean()

        # Save the heuristic
        avg_cover_type.to_pickle(HEURISTIC_FILE)
        
        self.metric = avg_cover_type
        
    def predict(self, test_data):
        
        # Get cover type labels
        mapping = self.metric.index.values
        label = np.vectorize(lambda x: mapping[x])
        
        # Calculate the difference between the test data and the heuristic
        diff_df = np.abs(self.metric.values - test_data.to_numpy()[:, None, :])

        # Sum the differences for each cover type
        ranking_df = diff_df.sum(axis=2)

        # Get the cover type with the lowest difference
        preds = np.argmin(ranking_df, axis=1)

        return label(preds)

