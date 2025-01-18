from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy import stats
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np

    
class parameterHT:
    def __init__(self):
        self.model_type = None
        self.params = {}

    def run(self):
        if self.model_type == 'Linear Regression':
            self.linear_regression_params()
        elif self.model_type == 'Lasso Regression':
            self.lasso_regression_params()
        elif self.model_type == 'RandomForest Regression':
            self.randomforest_regression_params()
        elif self.model_type == 'Decision Tree Regression':
            self.decision_tree_regression_params()
        elif self.model_type == 'XGBoost Regression':
            self.xgboost_regression_params()
        elif self.model_type == 'Support Vector Regression':
            self.svr_params()
        elif self.model_type == 'K-Neighbors Regression':
            self.knn_params()
        elif self.model_type == 'Neural Network Regression':
            self.mlp_params()
        elif self.model_type == 'Gaussian Process Regression':
            self.gpr_params()
        elif self.model_type == 'Ridge Regression':
            self.ridge_params()

    def linear_regression_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])

            with col1:
                par1 = st.checkbox("fit_intercept")
                fit_intercept_bool = None
                if par1:
                    fit_intercept_bool = st.radio("fit_intercept", ["True", "False"])
                    par1_value = True if fit_intercept_bool == "True" else False

            with col2:
                par2 = st.checkbox("Copy_X")
                copy_x = None
                if par2:
                    copy_x = st.radio("Copy_X", ["True", "False"])
                    par2_value = True if copy_x == "True" else False

            with col3:
                par3 = st.checkbox("Positive")
                positive = None
                if par3:
                    positive = st.radio("Positive", ["True", "False"])
                    par3_value = True if positive == "True" else False

            with col4:
                par4 = st.checkbox("n_jobs")
                n_jobs = None
                if par4:
                    n_jobs = st.number_input("Enter n_jobs:", value=2)
                    par4_value = n_jobs

        with container2:
            selected_params = []
            if fit_intercept_bool is not None:
                selected_params.append(f"fit_intercept: {fit_intercept_bool}")
                self.params["fit_intercept"] = [par1_value]
            if copy_x is not None:
                selected_params.append(f"Copy_X: {copy_x}")
                self.params["copy_X"] = [par2_value]
            if positive is not None:
                selected_params.append(f"Positive: {positive}")
                self.params["positive"] = [par3_value]
            if n_jobs is not None:
                selected_params.append(f"n_jobs: {n_jobs}")
                self.params["n_jobs"] = [par4_value]
            # Add params to the dictionary
            st.write(" | ".join(selected_params))

    def lasso_regression_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])

            with col1:
                par1 = st.checkbox("alpha")
                alpha = None
                if par1:
                    alpha = st.number_input("Enter alpha value", value=1.0)
                    par1_value = alpha

            with col2:
                par2 = st.checkbox("fit_intercept")
                fit_intercept = None
                if par2:
                    fit_intercept = st.radio("fit_intercept", ["True", "False"])
                    par2_value = True if fit_intercept == "True" else False

            with col3:
                par3 = st.checkbox("max_iter")
                max_iter = None
                if par3:
                    max_iter = st.number_input("Enter max_iter", value=1000)
                    par3_value = max_iter

            with col4:
                par4 = st.checkbox("tol")
                tol = None
                if par4:
                    tol = st.number_input("Enter tol value", value=1e-4)
                    par4_value = tol

        with container2:
            selected_params = []
            if alpha is not None:
                selected_params.append(f"alpha: {alpha}")
                self.params["alpha"] = [par1_value]
            if fit_intercept is not None:
                selected_params.append(f"fit_intercept: {fit_intercept}")
                self.params["fit_intercept"] = [par2_value]
            if max_iter is not None:
                selected_params.append(f"max_iter: {max_iter}")
                self.params["max_iter"] = [par3_value]
            if tol is not None:
                selected_params.append(f"tol: {tol}")
                self.params["tol"] = [par4_value]
            # st.write(self.params)
            st.write(" | ".join(selected_params))
            

    def randomforest_regression_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            colA, colB = st.columns([0.5, 0.5])

            with colA:
                n_estimators = st.number_input("n_estimators", min_value=10, max_value=500, value=100)
                max_depth = st.number_input("max_depth", min_value=1, max_value=50, value=10)
                min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=20, value=2)
                min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=20, value=1)

            with colB:
                max_features = st.selectbox("max_features", options=["auto", "sqrt", "log2"])
                bootstrap = st.radio("bootstrap", ["True", "False"])
                par_bootstrap = True if bootstrap == "True" else False
                random_state = st.number_input("random_state", value=42)
                n_jobs = st.number_input("n_jobs", min_value=1, max_value=20, value=1)
        
        # Display all selected parameters in a single line
        with container2:
            selected_params = [
                f"n_estimators: {n_estimators}",
                f"max_depth: {max_depth}",
                f"min_samples_split: {min_samples_split}",
                f"min_samples_leaf: {min_samples_leaf}",
                f"max_features: {max_features}",
                f"bootstrap: {bootstrap}",
                f"random_state: {random_state}",
                f"n_jobs: {n_jobs}",
            ]

            self.params["n_estimators"] = [n_estimators]
            self.params["max_depth"] = [max_depth]
            self.params["min_samples_split"] = [min_samples_split]
            self.params["min_samples_leaf"] = [min_samples_leaf]
            self.params["max_features"] = [max_features]
            self.params["bootstrap"] = [par_bootstrap]
            self.params["random_state"] = [random_state]
            self.params["n_jobs"] = [n_jobs]
            # Combine and display parameters
            # st.write(self.params)
            st.write(" | ".join(selected_params))

    def decision_tree_regression_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])

            with col1:
                par1 = st.checkbox("max_depth")
                max_depth = None
                if par1:
                    max_depth = st.number_input("Enter max_depth", value=10)

            with col2:
                par2 = st.checkbox("min_samples_split")
                min_samples_split = None
                if par2:
                    min_samples_split = st.number_input("min_samples_split", value=2)

            with col3:
                par3 = st.checkbox("min_samples_leaf")
                min_samples_leaf = None
                if par3:
                    min_samples_leaf = st.number_input("min_samples_leaf", value=1)

            with col4:
                par4 = st.checkbox("max_features")
                max_features = None
                if par4:
                    max_features = st.selectbox("max_features", options=['auto', 'sqrt', 'log2'])

        with container2:
            selected_params = []
            if max_depth is not None:
                selected_params.append(f"max_depth: {max_depth}")
                self.params["max_depth"] = [max_depth]
                
            if min_samples_split is not None:
                selected_params.append(f"min_samples_split: {min_samples_split}")
                self.params["min_samples_split"] = [min_samples_split]

            if min_samples_leaf is not None:
                selected_params.append(f"min_samples_leaf: {min_samples_leaf}")
                self.params["min_samples_leaf"] = [min_samples_leaf]

            if max_features is not None:
                selected_params.append(f"max_features: {max_features}")
                self.params["max_features"] = [max_features]

            # st.write(self.params)
            st.write(" | ".join(selected_params))

    def xgboost_regression_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])

            with col1:
                par1 = st.checkbox("n_estimators")
                n_estimators = None
                if par1:
                    n_estimators = st.number_input("n_estimators", value=100)

            with col2:
                par2 = st.checkbox("max_depth")
                max_depth = None
                if par2:
                    max_depth = st.number_input("max_depth", value=6)

            with col3:
                par3 = st.checkbox("learning_rate")
                learning_rate = None
                if par3:
                    learning_rate = st.number_input("learning_rate", value=0.1)

            with col4:
                par4 = st.checkbox("subsample")
                subsample = None
                if par4:
                    subsample = st.number_input("subsample", value=1.0)

        with container2:
            selected_params = []
            if n_estimators is not None:
                selected_params.append(f"n_estimators: {n_estimators}")
                self.params["n_estimators"] = [n_estimators]
            if max_depth is not None:
                selected_params.append(f"max_depth: {max_depth}")
                self.params["max_depth"] = [max_depth]
            if learning_rate is not None:
                selected_params.append(f"learning_rate: {learning_rate}")
                self.params["learning_rate"] = [learning_rate]
            if subsample is not None:
                selected_params.append(f"subsample: {subsample}")
                self.params["subsample"] = [subsample]
            st.write(self.params)
            st.write(" | ".join(selected_params))

    def svr_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2 = st.columns(2)

            with col1:
                par1 = st.checkbox("C")
                c_value = None
                if par1:
                    c_value = st.number_input("Enter C value", value=1.0)

            with col2:
                par2 = st.checkbox("kernel")
                kernel = None
                if par2:
                    kernel = st.selectbox("Select kernel", options=["linear", "poly", "rbf", "sigmoid"])

        with container2:
            selected_params = []
            if c_value is not None:
                selected_params.append(f"C: {c_value}")
                self.params["C"] = [c_value]
            if kernel is not None:
                selected_params.append(f"kernel: {kernel}")
                self.params["kernel"] = [kernel]
            st.write(" | ".join(selected_params))

    def knn_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2 = st.columns(2)

            with col1:
                par1 = st.checkbox("n_neighbors")
                n_neighbors = None
                if par1:
                    n_neighbors = st.number_input("Enter n_neighbors", value=5)

            with col2:
                par2 = st.checkbox("weights")
                weights = None
                if par2:
                    weights = st.selectbox("Select weights", options=["uniform", "distance"])

        with container2:
            selected_params = []
            if n_neighbors is not None:
                selected_params.append(f"n_neighbors: {n_neighbors}")
                self.params["n_neighbors"] = [n_neighbors]
            if weights is not None:
                selected_params.append(f"weights: {weights}")
                self.params["weights"] = [weights]
            st.write(" | ".join(selected_params))

    def mlp_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2 = st.columns(2)

            with col1:
                par1 = st.checkbox("hidden_layer_sizes")
                hidden_layer_sizes = None
                if par1:
                    hidden_layer_sizes = st.text_input("Enter hidden_layer_sizes", value="100")

            with col2:
                par2 = st.checkbox("activation")
                activation = None
                if par2:
                    activation = st.selectbox("Select activation", options=["identity", "logistic", "tanh", "relu"])

        with container2:
            selected_params = []
            if hidden_layer_sizes is not None:
                selected_params.append(f"hidden_layer_sizes: {hidden_layer_sizes}")
                self.params["hidden_layer_sizes"] = [tuple(map(int, hidden_layer_sizes.split(",")))]
            if activation is not None:
                selected_params.append(f"activation: {activation}")
                self.params["activation"] = [activation]
            st.write(" | ".join(selected_params))

    def gpr_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2 = st.columns(2)

            with col1:
                par1 = st.checkbox("alpha")
                alpha = None
                if par1:
                    alpha = st.number_input("Enter alpha", value=1e-10)

            with col2:
                par2 = st.checkbox("optimizer")
                optimizer = None
                if par2:
                    optimizer = st.selectbox("Select optimizer", options=["fmin_l_bfgs_b", "None"])

        with container2:
            selected_params = []
            if alpha is not None:
                selected_params.append(f"alpha: {alpha}")
                self.params["alpha"] = [alpha]
            if optimizer is not None:
                selected_params.append(f"optimizer: {optimizer}")
                self.params["optimizer"] = [optimizer]
            st.write(" | ".join(selected_params))

    def ridge_params(self):
        container1 = st.container()
        container2 = st.container()
        with container1:
            col1, col2 = st.columns(2)

            with col1:
                par1 = st.checkbox("alpha")
                alpha = None
                if par1:
                    alpha = st.number_input("Enter alpha", value=1.0)

            with col2:
                par2 = st.checkbox("fit_intercept")
                fit_intercept = None
                if par2:
                    fit_intercept = st.radio("Select fit_intercept", options=["True", "False"])

        with container2:
            selected_params = []
            if alpha is not None:
                selected_params.append(f"alpha: {alpha}")
                self.params["alpha"] = [alpha]
            if fit_intercept is not None:
                selected_params.append(f"fit_intercept: {fit_intercept}")
                self.params["fit_intercept"] = [True if fit_intercept == "True" else False]
            st.write(" | ".join(selected_params))


class getparameterHT:
    def __init__(self):
        self.model_type = None
        self.param_grid = {}

    def run(self):
        if self.model_type == 'Linear Regression':
            self.linear_regression_params()
        elif self.model_type == 'Lasso Regression':
            self.lasso_regression_params()
        elif self.model_type == 'Ridge Regression':
            self.ridge_regression_params()
        elif self.model_type == 'RandomForest Regression':
            self.randomforest_regression_params()
        elif self.model_type == 'Decision Tree Regression':
            self.decision_tree_regression_params()
        elif self.model_type == 'XGBoost Regression':
            self.xgboost_regression_params()
        elif self.model_type == 'Gradient Boosting Regression':
            self.gradient_boosting_params()
        elif self.model_type == 'AdaBoost Regression':
            self.adaboost_params()
        elif self.model_type == 'Support Vector Regression':
            self.svr_params()
        elif self.model_type == 'K-Neighbors Regression':
            self.kneighbors_params()
        elif self.model_type == 'Neural Network Regression':
            self.mlp_params()
        elif self.model_type == 'Gaussian Process Regression':
            self.gaussian_process_regression_params()

    def linear_regression_params(self):
        self.param_grid = {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'positive': [True, False],
            'n_jobs': np.arange(1, 20, 1)
        }

    def lasso_regression_params(self):
        self.param_grid = {
            'alpha': np.logspace(-4, 4, 50),
            'fit_intercept': [True, False],
            'max_iter': np.arange(100, 2000, 100)
        }

    def ridge_regression_params(self):
        self.param_grid = {
            'alpha': np.logspace(-4, 4, 50),
            'fit_intercept': [True, False],
            'max_iter': np.arange(100, 2000, 100),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }

    def randomforest_regression_params(self):
        self.param_grid = {
            'n_estimators': np.arange(10, 500, 10),
            'max_depth': np.arange(1, 50, 1),
            'min_samples_split': np.arange(2, 20, 1),
            'min_samples_leaf': np.arange(1, 20, 1),
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False],
            'n_jobs': np.arange(1, 20, 1)
        }

    def decision_tree_regression_params(self):
        self.param_grid = {
            'max_depth': np.arange(1, 50, 1),
            'min_samples_split': np.arange(2, 20, 1),
            'min_samples_leaf': np.arange(1, 20, 1),
            'max_features': ['auto', 'sqrt', 'log2']
        }

    def xgboost_regression_params(self):
        self.param_grid = {
            'n_estimators': np.arange(50, 1000, 50),
            'max_depth': np.arange(3, 20, 1),
            'learning_rate': np.logspace(-3, 0, 50),
            'subsample': np.linspace(0.5, 1.0, 50)
        }

    def gradient_boosting_params(self):
        self.param_grid = {
            'n_estimators': np.arange(50, 500, 50),
            'learning_rate': np.logspace(-3, 0, 50),
            'max_depth': np.arange(3, 20, 1),
            'min_samples_split': np.arange(2, 20, 1),
            'min_samples_leaf': np.arange(1, 20, 1)
        }

    def adaboost_params(self):
        self.param_grid = {
            'n_estimators': np.arange(50, 500, 50),
            'learning_rate': np.logspace(-3, 1, 50),
            'loss': ['linear', 'square', 'exponential']
        }

    def svr_params(self):
        self.param_grid = {
            'C': np.logspace(-3, 3, 50),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': np.arange(2, 5, 1),
            'gamma': ['scale', 'auto']
        }

    def kneighbors_params(self):
        self.param_grid = {
            'n_neighbors': np.arange(1, 50, 1),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': np.arange(10, 100, 10),
            'p': [1, 2]
        }

    def mlp_params(self):
        self.param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': np.logspace(-5, 3, 50),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': np.arange(100, 1000, 100)
        }
    
    def gaussian_process_regression_params(self):
        self.param_grid = {
            'alpha': np.logspace(-10, -1, 50),
            'kernel': [None, 'RBF', 'DotProduct', 'Matern'],
            'n_restarts_optimizer': np.arange(0, 10, 1),
            'normalize_y': [True, False]
        }

    def get_param_grid(self):
        return self.param_grid
    

def select_model(model_type, random_state=42):
    if model_type == 'Linear Regression':
        return LinearRegression()
    elif model_type == 'Lasso Regression':
        return linear_model.LassoLars(random_state=random_state)
    elif model_type == 'Ridge Regression':
        return Ridge()
    elif model_type == 'RandomForest Regression':
        return RandomForestRegressor(random_state=random_state)
    elif model_type == 'Decision Tree Regression':
        return DecisionTreeRegressor(random_state=random_state)
    elif model_type == 'XGBoost Regression':
        return XGBRegressor(random_state=random_state)
    elif model_type == 'Gradient Boosting Regression':
        return GradientBoostingRegressor(random_state=random_state)
    elif model_type == 'AdaBoost Regression':
        return AdaBoostRegressor(random_state=random_state)
    elif model_type == 'Support Vector Regression':
        return SVR()
    elif model_type == 'K-Neighbors Regression':
        return KNeighborsRegressor()
    elif model_type == 'Neural Network Regression':
        return MLPRegressor(random_state=random_state)
    elif model_type == 'Gaussian Process Regression':
        return GaussianProcessRegressor(random_state=random_state)
    else:
        return None
    
# Function to run RandomizedSearchCV
def perform_random_search(model, param_grid, X, y, selected_method='RandomSearch', random_state = None):
    if selected_method == 'RandomSearch':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=5, verbose=1, random_state=random_state, n_jobs=-1)
    elif selected_method == 'GridSearch':
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    search.fit(X, y)
    return search

