import streamlit as st
import apps.modelBuilding.module.visualize as visualization
import apps.modelBuilding.module.fetch_module as read_module
import apps.modelBuilding.module.analyze as module_analyze
import apps.modelBuilding.module.hyperparameter as hpt_tuning
from apps.modelBuilding.app.module import BestModel
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import io
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

 
class ModelBuilding:
    def __init__(self):
        self.title = "Model Building"
        self.X = None
        self.y = None
        self.model_type = None
        

    def run(self):
        st.title(self.title)
        col1, col2 = st.columns([0.8, 0.2])

        with col2:
            type_data = st.selectbox('Data Format :',('.xlsx','.parquet','.geojson'))

        with col1:
            uploaded_files = st.file_uploader("", accept_multiple_files=True)
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                try:
                    dataframe = read_module.read_data(uploaded_file, type_data)
                    st.session_state['dataset_train'] = dataframe
                except:
                    st.text("Can't read format data")


        try:
            model_method = ['Linear Regression','Lasso Regression','Ridge Regression','RandomForest Regression',
                            'Decision Tree Regression','XGBoost Regression','Gradient Boosting Regression',
                            'K-Neighbors Regression',
                            'Neural Network Regression','Gaussian Process Regression','AdaBoost Regression']

            column_numeric = module_analyze.getNumeric(st.session_state['dataset_train'])[0]
            df_numeric = st.session_state['dataset_train'][column_numeric]

            column_unique = module_analyze.getObject(st.session_state['dataset_train'])

            col_target = st.selectbox('Target :', tuple(column_numeric))
            
            column_non_target = []
            for i in column_numeric:
                if i not in col_target:
                    column_non_target.append(i)
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
                                v_train = st.number_input("Percentage Train", min_value=1, max_value=90, value = value)
                                v_test = st.number_input("Percentage Test", min_value=0, max_value=100-v_train, value=100-v_train)
                    random_state = st.checkbox("Randomstate")
                    rdm_st = None
                    if random_state:
                        rdm_st = st.number_input(" ", min_value=1, value = 42)
                    else :
                        rdm_st = 42
                    
                    
                with col3:
                    if show_message:
                        col_input = st.multiselect("Attribute Column(s):",column_non_target, default=column_non_target)
                    else:
                        col_input = st.multiselect("Attribute Column(s):",column_non_target)
                    if len(col_input)==0:
                        st.text("No Input Variable")
            
                data = df_numeric[[col_target]+col_input].fillna(0)

                

                if len(col_input)>0:
                    if st.button('Generate ML'):
                        try:
                            if outlier:
                                data = module_analyze.handling_oulier(data, outlier_method)
                            

                            if norm:
                                self.X = module_analyze.normalize_target(data, col_input)
                                self.y = data[col_target]
                            else:
                                self.X = data[col_input]
                                self.y = data[col_target]

                            if traintest:
                                train_size = round(v_train/100,2)
                                self.X, X_test, self.y, y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=rdm_st)

                            list_voi = []
                            for index, method in enumerate(model_method):
                                print(f"Success {method}")
                                
                                with st.expander(method):
                                    col1, col2 = st.columns([0.5,0.5])
                                    with col1:
                                        st.text("Base Model")
                                        model = hpt_tuning.select_model(method, rdm_st)
                                        if model is not None:
                                            model.fit(self.X, self.y)
                                        if traintest:

                                            df_eval = pd.DataFrame({
                                                'RMSE': [module_analyze.evaluate(model, self.X, self.y)[0].round(),
                                                            module_analyze.evaluate(model, X_test, y_test)[0].round()],
                                                'R2': [module_analyze.evaluate(model, self.X, self.y)[1].round(3),
                                                            module_analyze.evaluate(model, X_test, y_test)[1].round(3)],
                                                'MAE': [module_analyze.evaluate(model, self.X, self.y)[2].round(),
                                                            module_analyze.evaluate(model, X_test, y_test)[2].round()],
                                                'MAPE': [module_analyze.evaluate(model, self.X, self.y)[3].round(3),
                                                            module_analyze.evaluate(model, X_test, y_test)[3].round(3)],
                                            }).T
                                            df_eval.columns = ['Train', 'Test']

                                        else:
                                            print()
                                            df_eval = pd.DataFrame({
                                                'RMSE': [module_analyze.evaluate(model, self.X, self.y)[0].round()],
                                                'R2': [module_analyze.evaluate(model, self.X, self.y)[1].round(3)],
                                                'MAE': [module_analyze.evaluate(model, self.X, self.y)[2].round()],
                                                'MAPE': [module_analyze.evaluate(model, self.X, self.y)[3].round(3)]
                                            }).T
                                            df_eval.columns = ['error']
                                        
                                        st.dataframe(df_eval.reset_index().rename(columns={"index": 'Metrics'}))
  
                                        model_buffer = io.BytesIO()
                                        joblib.dump(model, model_buffer)
                                        model_buffer.seek(0)
                                        self.model_filename = f"{method.replace(' ', '_').lower()}_model.pkl"

                                        st.download_button(
                                            label="Save Model",
                                            data=model_buffer,
                                            file_name=self.model_filename,
                                            mime="application/octet-stream"
                                        )
                                
                            with st.expander("Best Model"):
                                best_model_instance = BestModel()
                                best_model_instance.X_train = self.X
                                best_model_instance.y_train = self.y
                                if traintest:
                                    best_model_instance.traintest = True
                                    best_model_instance.X_test = X_test
                                    best_model_instance.y_test = y_test

                                if random_state:
                                    best_model_instance.randomstate = rdm_st

                                model_results, model_, raw_model= best_model_instance.run()
                                st.text("Model performance:")
                                df_model = pd.DataFrame(raw_model)
                                if traintest:
                                    df_model.columns = ['Model','Cross Validation','Train MAE','Test MAE','Train MAPE','Test MAPE','Train R2','Test R2','Train RMSE','Test RMSE']
                                else:
                                    df_model.columns = ['Model','Cross Validation','MAE','MAPE','R2','RMSE']

                                if df_model.shape[0]>0:
                                    col1, col2 = st.columns([0.5,0.5])
                                    with col1:
                                        st.dataframe(df_model)
                                        best_model_name = model_results[0]['model']
                                        st.text(f"Best Model : {best_model_name}")
                                    with col2:
                                        df_voi = module_analyze.getCoefTable(model_[best_model_name].fit(self.X, self.y), best_model_name, self.X)
                                        df_voi['abs'] = df_voi['score'].abs()
                                        df_voi = df_voi.sort_values('abs', ascending=False).drop_duplicates('feature')
                                        fig = visualization.voi_chart(df_voi)
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.text("No model defined")


                        except:
                            st.text('Failed to generate')

        except:
            st.text('No Input Data')
        
def display():
    page = ModelBuilding()
    page.run()