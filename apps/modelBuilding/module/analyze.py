import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, normalize
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
 

def modeling(model_type, X, y, random_state= None):
    if model_type == 'Linear Regression':
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    elif model_type == 'Lasso Regression':
        model = linear_model.LassoLars(random_state=random_state)
        model.fit(X, y)
        return model
    
    elif model_type == 'RandomForest Regression':
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X, y)
        return model
    
    elif model_type == 'Decision Tree Regression':
        model = DecisionTreeRegressor(random_state=random_state)
        model.fit(X, y)
        return model
    
    elif model_type == 'XGBoost Regression':
        model = XGBRegressor(random_state=random_state)
        model.fit(X, y)
        return model
        
def getNumeric(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    dataframe_numeric = df.select_dtypes(include=numerics)
    column_numeric = dataframe_numeric.columns.tolist()
    return column_numeric, dataframe_numeric

def getObject(df):
    unique = df.nunique().reset_index()
    column_unique1 = unique[unique[0]==df.shape[0]]['index'].tolist()
    dataframe_object = df.select_dtypes(include=['object']).columns.tolist()
    column_unique = []
    for i in column_unique1:
        if i in dataframe_object:
            column_unique.append(i)
    return column_unique

def map_correlation(corr_value):
    if corr_value >= 0.5:
        return 'Strong Positive Correlation : > 0.5', '#073467'
    elif corr_value <= -0.5:
        return 'Strong Negative Correlation : < -0.5', '#700321'
    elif 0.2 <= corr_value < 0.5:
        return 'Moderate Positive Correlation : 0.2 to 0.5', '#365f96'
    elif -0.5 < corr_value <= -0.2:
        return 'Moderate Negative Correlation : -0.5 to -0.2', '#9f405a'
    elif 0.1 <= corr_value < 0.2:
        return 'Low Positive Correlation : 0.1 to 0.2', '#6689c6'
    elif -0.2 < corr_value <= -0.1:
        return 'Low Negative Correlation : -0.2 to -0.1', '#cd7e94'
    else:
        return 'Very low : -0.1 to 0.1', '#F0EDE5'

def getCorrelationdf(df_numeric, corr_, method):
    correlation = df_numeric.corr(method = method).round(2)
    correlation = correlation[~correlation.index.isin([corr_])]
    correlation_ = correlation.sort_values(corr_ , ascending = True).reset_index()
    correlation_[['Keterangan', 'Color']] = correlation_[corr_].apply(
                lambda x: pd.Series(map_correlation(x))
            )
    return correlation_


def normalize_target(df, col_feature):
    X = df[col_feature]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalizing the Data
    X_normalized = normalize(X_scaled)
    
    # Converting the numpy array into a pandas DataFrame
    X_normalized = pd.DataFrame(X_normalized)
    X_normalized.columns = X.columns
    return X_normalized

def handling_oulier(df, method):
    if method == 'ZScore':
        filtered_entries = np.array([True] * len(df))
        for col in df.columns.tolist():
            zscore = abs(stats.zscore(df[col])) # hitung absolute z-scorenya
            filtered_entries = (zscore < 3) & filtered_entries # keep yang kurang dari 3 absolute z-scorenya
            
        df = df[filtered_entries] # filter, cuma ambil yang z-scorenya dibawah 3
        return df
    elif method == 'IQR':
        filtered_entries = np.array([True] * len(df))
        for col in df.columns.tolist():
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            low_limit = Q1 - (IQR * 1.5)
            high_limit = Q3 + (IQR * 1.5)

            filtered_entries = ((df[col] >= low_limit) & (df[col] <= high_limit)) & filtered_entries
            
        df = df[filtered_entries]
        return df
    

def evaluate(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    return mse, r2, mae, mape

def evaluate_classifier(model, X, y):
    y_pred = model.predict(X)
    return {'accuracy': accuracy_score(y, y_pred), 
            'precision': precision_score(y, y_pred, average = 'micro'),
            'recall':recall_score(y,y_pred,average = 'micro'),
            'f1':f1_score(y, y_pred, average = 'micro')}

    
def getCoefTable(model, model_type, X):
    if model_type in ['Linear Regression', 'Lasso Regression', 'Ridge Regression']:
        importance = pd.DataFrame({
            'feature': X.columns.tolist(),
            'score': model.coef_
        })
        
        importance['abs'] = abs(importance['score'])
        importance['Color'] = np.where(importance['score'] < 0, '#be3455', 'lightblue')
        importance['Sign'] = np.where(importance['score'] < 0, 'negative', 'positive')
        return importance
    else:
        if model_type in ['RandomForest Regression','Decision Tree Regression',
                          'XGBoost Regression','Gradient Boosting Regression',]:
            shap_ex = shap.TreeExplainer(model)
        # elif model_type in ['Neural Network Regression', 'Gaussian Process Regression']:
        #     shap_ex = shap.DeepExplainer(model, X)
        else:
            shap_ex = shap.KernelExplainer(model.predict, X)

        shap_values = shap_ex.shap_values(X, check_additivity=False)
        shap_v = pd.DataFrame(shap_values, columns=X.columns)
        df = pd.DataFrame(data=X.values, columns=X.columns).reset_index(drop=True)

        corr_list = []
        for feature in shap_v.columns:
            corr = np.corrcoef(shap_v[feature], df[feature])[1][0]
            corr_list.append(corr)

        corr_df = pd.DataFrame({
            'feature': X.columns,
            'score': corr_list
        }).fillna(0)

        corr_df['Color'] = np.where(corr_df['score'] < 0, '#be3455', 'lightblue')
        corr_df = corr_df[corr_df['score'] != 0]
        corr_df = corr_df.sort_values('score', ascending=False)

        shap_abs = np.abs(shap_v)
        shap_mean = pd.DataFrame(shap_abs.mean(), columns=['abs']).reset_index()
        shap_mean.rename(columns={'index': 'feature'}, inplace=True)

        importance = shap_mean.merge(corr_df, on='feature', how='inner')
        importance['Sign'] = np.where(importance['score'] < 0, 'negative', 'positive')
        return importance
    


