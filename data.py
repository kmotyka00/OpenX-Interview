import pandas as pd
from sklearn.model_selection import train_test_split

def get_column_names():
    # Get the column names from the UCI website
    column_names = ["Elevation",
                    "Aspect",
                    "Slope", "Horizontal_Distance_To_Hydrology",
                    "Vertical_Distance_To_Hydrology",
                    "Horizontal_Distance_To_Roadways",
                    "Hillshade_9am",
                    "Hillshade_Noon",
                    "Hillshade_3pm",
                    "Horizontal_Distance_To_Fire_Points",
                    ]

    for i in range(1, 5):
        column_names.append(f"Wilderness_Area_{i}")
    for i in range(1, 41):
        column_names.append(f"Soil_Type_{i}")

    column_names.append("Cover_Type")
    return column_names

def load_data(url: str, column_names=None):
    # Load the data from the url
    df = pd.read_csv(url, delimiter=',', header=None)

    # Set the column names
    if column_names:
        df.columns = column_names

    return df

def preprocess_data(df):
    # Normalize the first 10 columns
    normalized_df = df
    first_10_columns = normalized_df.loc[:,
                                         'Elevation':'Horizontal_Distance_To_Fire_Points']
    normalized_df.loc[:, 'Elevation':'Horizontal_Distance_To_Fire_Points'] = (
        first_10_columns-first_10_columns.min())/(first_10_columns.max()-first_10_columns.min())

    return normalized_df

def split_data(df):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('Cover_Type', axis=1), df['Cover_Type'], test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

df = load_data(url='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
               column_names=get_column_names())

normalized_df = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(normalized_df)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)