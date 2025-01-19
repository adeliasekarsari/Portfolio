import streamlit as st
from hydralit import HydraHeadApp
from apps.project.module.module import get_base64_image
from apps.modelBuilding.main import ModelBuildingApp

class AutoMLApp(HydraHeadApp):
    def __init__(self, title='Automation Machine', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        # Display content for EstehApp
        st.title(self.title)

        # Add a back button
        if st.button("Back to Projects"):
            # Set session state to navigate back to the project app
            st.session_state.selected_project = None

        if st.session_state.get("show_automl_app", False):
            ModelBuildingApp().run()  # Call SimilarityApp's run method
            return 
        
        if st.button("Hands-On AutoML Apps"):
            st.session_state["show_automl_app"] = True


        st.markdown("""
        This AutoML Regression platform is a powerful tool designed to simplify the process of creating predictive models 
                    for regression tasks. Users can upload their dataset, define the target variable and attributes, and configure
                     essential preprocessing steps such as normalization, outlier handling, train-test splitting, and setting a
                     random state for reproducibility. The platform evaluates 12 different regression models, including Linear Regression, 
                    Lasso Regression, Ridge Regression, Random Forest Regression, and XGBoost Regression, among others. It calculates metrics such as RMSE, MAE, MAPE, 
                    and R² for each model and determines the best-performing one based on user-defined thresholds. Additionally, it provides a 
                    summary of the results and visualizations, such as variable importance for the top model, helping users make informed decisions with minimal 
                    effort and technical expertise.""")
        
        with st.container():
            col1, col2 = st.columns([0.65,0.35])
            with col1:
                img = get_base64_image(f"apps/project/style/automl/main.png")
                st.markdown(
                    f"""
                    <div class="card">
                        <img src="{img}" style="width: 100%; height: auto;" alt="Project Image">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.header("")
                st.markdown("""
                To get started, users need to upload their dataset, which can be in Parquet, Excel, or GeoJSON format. 
                            After the dataset is loaded, they should specify the target column—the variable they wish to 
                            predict—and select the attribute columns, which represent the input features for the regression model. 
                            The platform offers advanced customization options, such as setting train-test split ratios, normalizing 
                            the data to standardize it, handling outliers to address extreme values, and defining a random state to 
                            ensure reproducibility. Once all configurations are set, users can click the "Generate" button to initiate 
                            the modeling process. The system evaluates the dataset against 12 regression models and generates key 
                            performance metrics for each. The results include detailed metrics, a summary of the best-performing model, 
                            and a visualization of the variable importance, enabling users to identify the model that best fits their needs.""")
                
        with st.container():
            st.markdown("## Result")
            col2, col1 = st.columns([0.3,0.7])
            with col1:
                result1 = get_base64_image(f"apps/project/style/automl/result1.png")
                st.markdown(
                    f"""
                    <div class="card">
                        <img src="{result1}" style="width: 100%; height: auto;" alt="Project Image">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown("""
                The results of the 12 regression models are displayed with performance metrics such as RMSE, MAE, MAPE, 
                            and R². These metrics help users evaluate the effectiveness of each model, enabling c
                            omparisons and highlighting differences in accuracy and prediction reliability.
                """)
                result2 = get_base64_image(f"apps/project/style/automl/result2.png")
                st.markdown(
                    f"""
                    <div class="card">
                        <img src="{result2}" style="width: 75%; height: auto;" alt="Project Image">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.header("")   
        with st.container():
            col1, col2 = st.columns([0.6,0.4])
            with col1:
                result3 = get_base64_image(f"apps/project/style/automl/result3.png")
                st.markdown(
                    f"""
                    <div class="card">
                        <img src="{result3}" style="width: 95%; height: auto;" alt="Project Image">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown("""
            A summary table compiles the performance metrics for all 12 models, making it easier to compare results and 
                        identify trends. The best-performing model is selected based on user-defined thresholds for metrics, 
                        ensuring an optimal choice for the given dataset. """)
                st.markdown("""
            The variable importance visualization showcases the influence of each feature in the dataset on the predictions made by 
                            the best model. This analysis provides valuable insights into the most impactful variables, helping users 
                            understand the drivers behind the model's predictions and refine their data strategies for future modeling.""")

        






