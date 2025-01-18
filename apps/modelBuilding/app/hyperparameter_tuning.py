import streamlit as st
import apps.modelBuilding.module.visualize as visualization
import apps.modelBuilding.module.fetch_module as read_module
import apps.modelBuilding.module.analyze as module_analyze
import pandas as pd
import apps.modelBuilding.module.hyperparameter as hpt_tuning
from sklearn.model_selection import train_test_split
import joblib
import io
import time
import os

class HyperparameterTuning:
    def __init__(self):
        self.title = "Hyperparameter Tuning Page"
        self.model_filename = None  

    def run(self):
        st.title(self.title)
        if 'second_button' not in st.session_state:
            st.session_state.second_button = False

        col1, col2 = st.columns([0.8, 0.2])

        with col2:
            type_data = st.selectbox('Data Format :',('.xlsx','.csv','.parquet','.geojson'))

        with col1:
            uploaded_files = st.file_uploader("", accept_multiple_files=True)
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                try:
                    dataframe = read_module.read_data(uploaded_file, type_data)
                    st.session_state['df'] = dataframe
                except:
                    st.text("Can't read format data")

        try:
            column_numeric = module_analyze.getNumeric(dataframe)[0]
            df_numeric = dataframe[column_numeric]
            column_unique = module_analyze.getObject(dataframe)

            try:
                col_target = st.selectbox('Target :', tuple(column_numeric))

                column_non_target = [i for i in column_numeric if i != col_target]
                col3, col4 = st.columns([0.85,0.15])

                with st.container():
                    with col4:
                        st.text("")
                        st.text("")
                        show_message = st.checkbox('All Attribute')
                        norm = st.checkbox("Data Normalization")
                        col1, col2 = st.columns([0.5,0.5])
                        with col1:
                            outlier = st.checkbox("Handling Outlier")
                        with col2:
                            if outlier:
                                outlier_method = st.radio("",['ZScore','IQR'])
                        col1, col2 = st.columns([0.5,0.5])
                        with col1:
                            traintest = st.checkbox("Split Train Test")
                            if traintest:
                                with col2:
                                    value = 70
                                    v_train = st.number_input("Percentage Train", min_value=1, max_value=99, value = value)
            
                        
                    with col3:
                        if show_message:
                            col_input = st.multiselect("Attribute Column(s):",column_non_target, default=column_non_target)
                        else:
                            col_input = st.multiselect("Attribute Column(s):",column_non_target)
                        if len(col_input)==0:
                            st.warning("No Input Variable")
                        else:
                            model_type = st.selectbox('Model :', ('Linear Regression',
                                                                    'Lasso Regression',
                                                                    'Ridge Regression',
                                                                    'RandomForest Regression',
                                                                    'Decision Tree Regression',
                                                                    'XGBoost Regression',
                                                                    'Support Vector Regression',
                                                                    'K-Neighbors Regression',
                                                                    'Neural Network Regression',
                                                                    'Gaussian Process Regression'
                                                ))
                        st.text("**Support Vector Regression, K-Neighbors Regression, Neural Network Regression and Gaussian Process Regression take several time to process")
                        ht_method = st.radio("",['Auto Hyperparameter Tuning','Manual Input Hyperparameter Tuning'],horizontal=True)
                        with st.container():
                            if ht_method == 'Manual Input Hyperparameter Tuning':
                                param = hpt_tuning.parameterHT()
                                param.model_type = model_type 
                                hyperparams = param.run()
                            
                    with col4:
                        rdm_st = None
                        if model_type != "Linear Regression":
                            random_state = st.checkbox("Randomstate")
                            if random_state:
                                rdm_st = st.number_input(" ", min_value=1, value = 42)

                data = df_numeric[[col_target]+col_input].fillna(0)
                if 'second_button' not in st.session_state:
                    st.session_state.second_button = False


                if len(col_input)>0:
                    search_method = st.radio("  ",['RandomSearch', 'GridSearch'],horizontal=True)
                    if st.button('Generate ML With Hyperparameter Tuning'):
                        try:
                            with st.spinner("Processing..."):
                                results_placeholder = st.empty()
                                with results_placeholder.container():
                                    st.header("Before Tuning")
                                    if outlier:
                                        data = module_analyze.handling_oulier(data, outlier_method)

                                    if norm:
                                        self.X = module_analyze.normalize_target(data, col_input)
                                    else:
                                        self.X = data[col_input]
                                    self.y = data[col_target]

                                    if traintest:
                                        self.X, X_test, self.y, y_test = train_test_split(self.X, self.y, train_size=v_train, random_state=42)

                                    model = hpt_tuning.select_model(model_type,  random_state=rdm_st)
                                    if model is not None:
                                        model.fit(self.X, self.y)


                                    if traintest:
                                        df_eval = pd.DataFrame({
                                            'RMSE': [module_analyze.evaluate(model, self.X, self.y)[0].round(3),
                                                        module_analyze.evaluate(model, X_test, y_test)[0].round(3)],
                                            'R2': [module_analyze.evaluate(model, self.X, self.y)[1].round(3),
                                                        module_analyze.evaluate(model, X_test, y_test)[1].round(3)],
                                            'MAE': [module_analyze.evaluate(model, self.X, self.y)[2].round(3),
                                                        module_analyze.evaluate(model, X_test, y_test)[2].round(3)],
                                            'MAPE': [module_analyze.evaluate(model, self.X, self.y)[3].round(3),
                                                        module_analyze.evaluate(model, X_test, y_test)[3].round(3)],
                                        }).T
                                        df_eval.columns = ['Train', 'Test']

                                    else:
                                        df_eval = pd.DataFrame({
                                            'RMSE': [module_analyze.evaluate(model, self.X, self.y)[0].round(3)],
                                            'R2': [module_analyze.evaluate(model, self.X, self.y)[1].round(3)],
                                            'MAE': [module_analyze.evaluate(model, self.X, self.y)[2].round(3)],
                                            'MAPE': [module_analyze.evaluate(model, self.X, self.y)[3].round(3)]
                                        }).T
                                        df_eval.columns = ['error']
                                    

                                    df_voi = module_analyze.getCoefTable(model, model_type, self.X)
                                    df_voi['abs'] = df_voi['score'].abs()
                                    df_voi = df_voi.sort_values('abs', ascending=False).drop_duplicates("feature")

                                    col3, col4 = st.columns([0.3, 0.7])
                                            
                                    with col3:
                                        st.dataframe(df_eval.reset_index().rename(columns={"index": 'Metrics'}))
                                        st.dataframe(df_voi[['feature', 'score']])
                                    
                                    with col4:
                                        fig = visualization.voi_chart(df_voi)
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                            

                                if ht_method == 'Auto Hyperparameter Tuning':
                                    with st.container():
                                        st.header("After Tuning")
                                        param_selector = hpt_tuning.getparameterHT()
                                        param_selector.model_type = model_type
                                        param_selector.run()
                                        param_grid = param_selector.get_param_grid()

                                        model = hpt_tuning.select_model(model_type, random_state = rdm_st)
                                        
                                        if model is not None:
                                            search_results = hpt_tuning.perform_random_search(model, param_grid, self.X, self.y, search_method, rdm_st)
                                            st.write("Best parameters found: ", search_results.best_params_)

                                            best_model = search_results.best_estimator_
                                            best_model.fit(self.X, self.y)

                                            if traintest:
                                                df_eval_tune = pd.DataFrame({
                                                        'RMSE': [module_analyze.evaluate(best_model, self.X, self.y)[0].round(3),
                                                                    module_analyze.evaluate(best_model, X_test, y_test)[0].round(3)],
                                                        'R2': [module_analyze.evaluate(best_model, self.X, self.y)[1].round(3),
                                                                    module_analyze.evaluate(best_model, X_test, y_test)[1].round(3)],
                                                        'MAE': [module_analyze.evaluate(best_model, self.X, self.y)[2].round(3),
                                                                    module_analyze.evaluate(best_model, X_test, y_test)[2].round(3)],
                                                        'MAPE': [module_analyze.evaluate(best_model, self.X, self.y)[3].round(3),
                                                                    module_analyze.evaluate(best_model, X_test, y_test)[3].round(3)],
                                                    }).T
                                                df_eval_tune.columns = ['Train', 'Test']
                                                

                                            else:
                                                df_eval_tune = pd.DataFrame({
                                                    'RMSE': [module_analyze.evaluate(best_model, self.X, self.y)[0].round(3)],
                                                    'R2': [module_analyze.evaluate(best_model, self.X, self.y)[1].round(3)],
                                                    'MAE': [module_analyze.evaluate(best_model, self.X, self.y)[2].round(3)],
                                                    'MAPE': [module_analyze.evaluate(best_model, self.X, self.y)[3].round(3)]
                                                }).T
                                                df_eval_tune.columns = ['error']
                                            
                                            df_voi_tune = module_analyze.getCoefTable(best_model, model_type, self.X)
                                            df_voi_tune['abs'] = df_voi_tune['score'].abs()
                                            df_voi_tune = df_voi_tune.sort_values('abs', ascending=False).drop_duplicates("feature")

                                            col3, col4 = st.columns([0.3, 0.7])
                                            if df_eval_tune.equals(df_eval):
                                                st.text("Cannot build model with Parameter selected")
                                            else:     
                                                with col3:
                                                    st.dataframe(df_eval_tune.reset_index().rename(columns={"index": 'Metrics'}))
                                                    st.dataframe(df_voi_tune[['feature', 'score']])
                                            
                                                with col4:
                                                    fig = visualization.voi_chart(df_voi_tune)
                                                    st.plotly_chart(fig, use_container_width=True)

                                            model_buffer = io.BytesIO()
                                            joblib.dump(best_model, model_buffer)
                                            model_buffer.seek(0)
                                            self.model_filename = f"{model_type.replace(' ', '_').lower()}_aftertuning_model.pkl"

                                            st.download_button(
                                                label="Save Model",
                                                data=model_buffer,
                                                file_name=self.model_filename,
                                                mime="application/octet-stream"
                                            )
                                elif ht_method == 'Manual Input Hyperparameter Tuning':
                                    with st.container():
                                        st.header("After Tuning")
                                        ht = param
                                        param_grid = ht.params
                                        if model is not None:
                                            search_results = hpt_tuning.perform_random_search(model, param_grid, self.X, self.y, search_method, rdm_st) 
                                            st.write("Parameters found: ", search_results.best_params_)
                                            if search_results.best_params_ == {}:
                                                st.write("No selected Param")
                                            else:
                                                best_model = search_results.best_estimator_
                                                best_model.fit(self.X, self.y)

                                                if traintest:
                                                    df_eval_tune = pd.DataFrame({
                                                        'RMSE': [module_analyze.evaluate(best_model, self.X, self.y)[0].round(3),
                                                                    module_analyze.evaluate(best_model, X_test, y_test)[0].round(3)],
                                                        'R2': [module_analyze.evaluate(best_model, self.X, self.y)[1].round(3),
                                                                    module_analyze.evaluate(best_model, X_test, y_test)[1].round(3)],
                                                        'MAE': [module_analyze.evaluate(best_model, self.X, self.y)[2].round(3),
                                                                    module_analyze.evaluate(best_model, X_test, y_test)[2].round(3)],
                                                        'MAPE': [module_analyze.evaluate(best_model, self.X, self.y)[3].round(3),
                                                                    module_analyze.evaluate(best_model, X_test, y_test)[3].round(3)],
                                                    }).T
                                                    df_eval_tune.columns = ['Train', 'Test']
                                                    

                                                else:
                                                    df_eval_tune = pd.DataFrame({
                                                        'RMSE': [module_analyze.evaluate(best_model, self.X, self.y)[0].round(3)],
                                                        'R2': [module_analyze.evaluate(best_model, self.X, self.y)[1].round(3)],
                                                        'MAE': [module_analyze.evaluate(best_model, self.X, self.y)[2].round(3)],
                                                        'MAPE': [module_analyze.evaluate(best_model, self.X, self.y)[3].round(3)]
                                                    }).T
                                                    df_eval_tune.columns = ['error']
                                                
                                                df_voi_tune = module_analyze.getCoefTable(best_model, model_type, self.X)
                                                df_voi_tune['abs'] = df_voi_tune['score'].abs()
                                                df_voi_tune = df_voi_tune.sort_values('abs', ascending=False).drop_duplicates("feature")

                                                col3, col4 = st.columns([0.3, 0.7])
                                                if df_eval_tune.equals(df_eval):
                                                    st.text("Cannot build model with Parameter selected")
                                                else:     
                                                    with col3:
                                                        st.dataframe(df_eval_tune.reset_index().rename(columns={"index": 'Metrics'}))
                                                        st.dataframe(df_voi_tune[['feature', 'score']])
                                                
                                                    with col4:
                                                        fig = visualization.voi_chart(df_voi_tune)
                                                        st.plotly_chart(fig, use_container_width=True)

                                                    st.session_state.second_button = True

                                                    model_buffer = io.BytesIO()
                                                    joblib.dump(best_model, model_buffer)
                                                    model_buffer.seek(0)
                                                    self.model_filename = f"{model_type.replace(' ', '_').lower()}_aftertuning_model.pkl"

                                                    st.download_button(
                                                        label="Save Model",
                                                        data=model_buffer,
                                                        file_name=self.model_filename,
                                                        mime="application/octet-stream"
                                                    )
                        except:
                            st.warning('Failed build VOI')

            except:
                st.warning('Failed build Model')
        except:
            st.warning('No Input Data')

        
def display():
    page = HyperparameterTuning()
    page.run()