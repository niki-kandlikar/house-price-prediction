"""
modeling.py
------------
Train/test split, linear regression fitting, and evaluation using
MSE and R^2, as called out on the resume.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    return {"mse": mse, "rmse": rmse, "r2": r2, "predictions": preds}


def coefficient_table(model, feature_names):
    return pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)
