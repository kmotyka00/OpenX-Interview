from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

def create_nn(X_train, y_train):
    def create_model(input_shape=(54,), num_hidden_layers=1, num_hidden_units=32, 
                    dropout_rate=0.2, learning_rate=0.001):
        
        model = Sequential()

        # Input layer
        model.add(layers.Dense(num_hidden_units, input_shape=input_shape, activation='relu'))

        # Hidden layers
        for i in range(num_hidden_layers):
            model.add(layers.Dense(num_hidden_units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))

        # Output layers
        model.add(layers.Dense(7, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        return model


    # Create a Keras classifier
    clf_nn = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=1)

    # Define the hyperparameters to search
    param_grid = {'num_hidden_layers': [1, 2, 3],
                'num_hidden_units': [32, 64, 128],
                'dropout_rate': [0.2, 0.3, 0.4],
                'learning_rate': [0.001, 0.01, 0.1]}
    
    # Perform the search
    grid_search = GridSearchCV(clf_nn, param_grid, cv=3, verbose=1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    
    # Train the best model
    best_model = grid_search.best_estimator_
    print(best_model)
    best_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[EarlyStopping(patience=3)])