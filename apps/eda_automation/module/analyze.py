import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, normalize
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


def getCorrelationdf(df_numeric, corr_):
    correlation = df_numeric.corr().round(2)
    correlation = correlation[~correlation.index.isin([corr_])]
    correlation_ = correlation.sort_values(corr_ , ascending = True).reset_index()
    correlation_['Keterangan'] = np.where(correlation_[corr_]>=0.5,'Strong Positive Correlation : > 0.5',
                                            np.where(correlation_[corr_]<=-0.5,'Strong Negative Correlation : < - 0.5',
                                                        np.where((correlation_[corr_]<0.5)&(correlation_[corr_]>=0.2), 'Moderate Positive Correlation : 0.2 to 0.5',
                                                                np.where((correlation_[corr_]<=-0.2)&(correlation_[corr_]>-0.5), 'Moderate Negative Correlation : -0.5 to -0.2',
                                                                        np.where((correlation_[corr_]<0.2)&(correlation_[corr_]>=0.1), 'Low Positive Correlation : 0.1 to 0.2',
                                                                                np.where((correlation_[corr_]<=-0.1)&(correlation_[corr_]>-0.2),'Low Negative Correlation : -0.2 to -0.1',
                                                                                            'Very low : -0.1 to 0.1')

                                                                ))))
                                            )
        
    correlation_["Color"] = np.where(correlation_[corr_]>=0.5,'#073467',
                                        np.where(correlation_[corr_]<=-0.5,'#700321',
                                                    np.where((correlation_[corr_]<0.5)&(correlation_[corr_]>=0.2), '#365f96',
                                                            np.where((correlation_[corr_]<=-0.2)&(correlation_[corr_]>-0.5), '#9f405a',
                                                                    np.where((correlation_[corr_]<0.2)&(correlation_[corr_]>=0.1), '#6689c6',
                                                                            np.where((correlation_[corr_]<=-0.1)&(correlation_[corr_]>-0.2),'#cd7e94','#F0EDE5')
                                                                                        )

                                                            ))))
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

def evaluate(model, X, y):
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred, squared=False), r2_score(y, y_pred)

def evaluate_classifier(model, X, y):
    y_pred = model.predict(X)
    return {'accuracy': accuracy_score(y, y_pred), 
            'precision': precision_score(y, y_pred, average = 'micro'),
            'recall':recall_score(y,y_pred,average = 'micro'),
            'f1':f1_score(y, y_pred, average = 'micro')}

    
def getCoefTable(model, model_type, X):
    if model_type == 'Linear Regression' or model_type == 'Lasso Regression':
        importance = pd.DataFrame({'feature':X.columns.tolist(),
                                'score':model.coef_
                                })
        importance['abs'] = abs(importance['score'])
        ## here I'm adding a column with colors
        importance["Color"] = np.where(importance["score"]<0, '#be3455', 'lightblue')
        importance['Sign'] = np.where(importance['score']<0,'negative','positive')
        return importance
    else:
        shap_ex = shap.TreeExplainer(model)
        shap_values = shap_ex.shap_values(X,check_additivity=False)

        shap_v = pd.DataFrame(shap_values)
        shap_v.columns = X.columns

        df = X.copy()
        df = pd.DataFrame(data = df,
                        columns=list(X.columns))
        df_v = df.reset_index(drop=True)

        corr_list = []
        for i in shap_v.columns:
            b = np.corrcoef(shap_v[i],df_v[i])[1][0]
            corr_list.append(b)
        corr_df = pd.concat([pd.Series(X.columns),pd.Series(corr_list)],axis=1).fillna(0)
        corr_df.columns  = ['feature','score']

        corr_df['Color'] = np.where(corr_df['score']<0,'#be3455','lightblue')
        corr_df = corr_df[corr_df['score']!=0]
        corr_df = corr_df.sort_values('score',ascending=False)
        # Plot it
        shap_abs = np.abs(shap_v)

        k=pd.DataFrame(shap_abs.mean()).reset_index()
        k.columns = ['feature','abs']
        importance = k.merge(corr_df,left_on = 'feature',right_on='feature',how='inner')
        importance['Sign'] = np.where(importance['score']<0,'negative','positive')
        return importance
    


