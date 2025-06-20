import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_train.csv")
    data_train = pd.read_csv(file_path)
    file_path_test = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_test.csv")
    data_test = pd.read_csv(file_path_test)

    X_train = data_train.drop("Attrition", axis=1)
    y_train = data_train["Attrition"]
    X_test = data_test.drop("Attrition", axis=1)
    y_test = data_test["Attrition"]

    input_example = X_train[0:5]
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    mlflow.sklearn.autolog()  # Move this up, before training
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Predict and log accuracy
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Explicitly log the model (this creates artifacts/model/)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)

