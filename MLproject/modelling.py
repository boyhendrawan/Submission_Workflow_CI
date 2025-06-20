import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, roc_auc_score, log_loss
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

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        mlflow.sklearn.autolog()  # Move this up, before training

        ## Log parameter
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='attrition_model',
            input_example=input_example
            )
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log evaluation metrics (sama dengan autolog)
        y_pred = model.predict(X_test)
        precission = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precission', precission)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1score)
        mlflow.log_metric('custom_roc_auc', auc)
        mlflow.log_metric('custom_log_loss', logloss)

