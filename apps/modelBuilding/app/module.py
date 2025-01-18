import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st


def evaluate_models(models, X_train, y_train):
    results = []

    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        mean_cv_mae = -np.mean(cv_scores)

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)

        # Metrics
        mae = mean_absolute_error(y_train, y_pred)
        mape = np.mean(np.abs((y_train - y_pred) / y_train)) * 100
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))

        mae = 0 if np.isinf(mae) or np.isnan(mae) else mae
        mape = 0 if np.isinf(mape) or np.isnan(mape) else mape
        r2 = 0 if np.isinf(r2) or np.isnan(r2) else r2
        rmse = 0 if np.isinf(rmse) or np.isnan(rmse) else rmse

        results.append({
            "model": name,
            "cv_mae": mean_cv_mae,
            "mae": mae,
            "mape": round(mape),
            "r2": r2,
            "rmse": round(rmse)
        })

    return results

def evaluate_models_split(models, X_train, y_train, X_test, y_test, test_size=0.2, random_state=42):
    # Split data into training and testing sets
    results = []

    for name, model in models.items():
        # Cross-validation on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        mean_cv_mae = -np.mean(cv_scores)

        # Train the model
        model.fit(X_train, y_train)

        # Predict on training and testing sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics for training data
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Metrics for testing data
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Append results
        results.append({
            "model": name,
            "cv_mae": round(mean_cv_mae),
            "train_mae": round(train_mae),
            "test_mae": round(test_mae),
            "train_mape": round(train_mape),
            "test_mape": round(test_mape),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": round(train_rmse),
            "test_rmse": round(test_rmse)
        })

    return results


class BestModel:
    def __init__(self, title='', **kwargs):
        self.__dict__.update(kwargs)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.traintest = False
        self.randomstate = 42

    def run(self):
        model_method = {
            'Linear Regression': LinearRegression(),
            'Lasso Regression': linear_model.LassoLars(random_state=self.randomstate),
            'RandomForest Regression': RandomForestRegressor(random_state=self.randomstate),
            'Decision Tree Regression': DecisionTreeRegressor(random_state=self.randomstate),
            'XGBoost Regression': XGBRegressor(random_state=self.randomstate),
            'Support Vector Regression': SVR(),
            'K-Neighbors Regression': KNeighborsRegressor(),
            'Neural Network Regression': MLPRegressor(random_state=self.randomstate),
            'Gaussian Process Regression': GaussianProcessRegressor(random_state=self.randomstate),
            'Ridge Regression': Ridge(),
            'AdaBoost Regression':AdaBoostRegressor(random_state=self.randomstate),
            'Gradient Boosting Regression':GradientBoostingRegressor(random_state=self.randomstate)
        }
        
        
        if self.traintest:
            model_ = evaluate_models_split(model_method, self.X_train, self.y_train, self.X_test, self.y_test)
            model_results = [res for res in model_ if (res['train_r2'] > 0 and res['train_r2']<=0.98 and res['test_r2']>0)]
            if len(model_results)<=1:
                model_results = [res for res in model_ if (res['train_r2'] > 0 and res['train_r2']<=0.98)]
                if len(model_results)<=0:
                    model_results = [res for res in model_ if (res['train_r2'] > 0)]
            sorted_results = sorted(
                model_results, 
                key=lambda x: (x['train_rmse'], -x['train_r2'], x['train_mae'], x['cv_mae'])
            )
        else:
            model_ = evaluate_models(model_method, self.X_train, self.y_train)
            model_results = [res for res in model_ if (res['r2'] >= 0 and res['r2']<=0.98)]
            if len(model_results)<=1:
                model_results = [res for res in model_ if (res['r2'] >= 0)]
            sorted_results = sorted(
                model_results, 
                key=lambda x: (x['rmse'], -x['r2'], x['mae'], x['cv_mae'])
            )
        return sorted_results, model_method, model_
