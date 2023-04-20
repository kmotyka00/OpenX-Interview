# OpenX-Interview
This repository contains solution for OpenX Machine Learning Intern 2023 task.

Directories description:

- Models: all of the models saved with grid search results and history of the neural network trening
- config_files: file with constants (relative paths to save models, dataset url etc.)
- data_preprocessing: file which includes functions to load and normalize dataset, split into train and test set etc.
- machine_learning: 3 files for simple heuristic, sklearn models and neural networks + evaluation app
- rest_api: REST API created using Flask and a file with helper functions

Additionaly there is a requirements.txt file with libraries needed to run the code and Dockerfile to create Docker image.

Heuristic description: for every cover type I calculate mean values of each features and then I use them to classify new data (lowest mean error between new data and calculated averages).

Sklearn models: baseline Decission Tree and Logistic Regression

Neural Network: parameters found by Grid Search: 
  - 2 hidden layers,
  - 128 units each,
  - Dropout 0.2,
  - Learning rate 0.01.

Accuracy on test set:

- Simple heuristic: 37,5 %
- Decission Tree: 93,9 %
- Logistic Regression: 72 %
- Neural Network: 88,7 %

[REST API] Sample request and response:
![image](https://user-images.githubusercontent.com/61949638/233479104-e581516c-1336-4467-a132-9880c5961e86.png)
