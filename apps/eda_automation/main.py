import streamlit as st
from streamlit_option_menu import option_menu
import apps.eda_automation.module.visualize as visualization
import apps.eda_automation.module.fetch_module as read_module
import apps.eda_automation.module.analyze as module_analyze
from hydralit import HydraHeadApp
import pandas as pd
from sklearn.linear_model import LinearRegression
from PIL import Image
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class EdaApp(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        with st.sidebar:
            choose = option_menu("EDA Automation", ["Project Description",
                                                    "Data Distribution",
                                                    "Correlation Analysis",
                                                    "Variable of Importance Analysis"],
                                icons=['book','pin-map-fill','person lines fill','book'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "black"},
                "icon": {"color": "White", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#6BA1D1"},
            }
            
            )
        
        if choose == "Project Description":
            st.markdown("""
            <style>
            .big-font {
                font-size:60px !important;
            }
            .medium-font {
                font-family:sans-serif;
                font-size:24px !important;
                text-align: justify color:White;
            }
            </style>
            """, unsafe_allow_html=True)


            # Title

            st.markdown("<p class='big-font'>Explore Your Data! </p>", unsafe_allow_html=True)
            

            st.title("")
            text = """
            The EDA Wizard is your go-to solution for automated Exploratory Data Analysis. 
            Quickly analyze datasets with built-in Correlation Analysis, Histogram Visualization, 
            and Variable Importance Assessment. Uncover insights effortlessly, visualize data 
            distributions, and identify influential variables. Simplify 
            your data exploration with this intuitive dashboard.
            """
            st.markdown(f"<p class='medium-font'>{text}</p>", unsafe_allow_html=True)

            st.title("")
            st.title("")
            image1 = Image.open(r".\apps\eda_automation\images\histogram.png")
            image2 = Image.open(r".\apps\eda_automation\images\correlation.png")
            image3 = Image.open(r".\apps\eda_automation\images\voi.png")

            # Add image with image desc
            with st.container():
                coll1, coll2, coll3 = st.columns([0.25,0.25,0.25])
                with coll1:
                    col1, col2, col3 = st.columns([0.15,0.8,0.1])
                    with col1:
                        st.write('')

                    with col2:
                        st.image(image1, width = 450)
                        

                    with col3:
                        st.write('')
                    
                    st.markdown('<p style="text-align: center;">Correlation Analysis</p>',unsafe_allow_html=True)
                with coll2:
                    col1, col2, col3 = st.columns([0.15,0.8,0.1])
                    with col1:
                        st.write('')

                    with col2:
                        st.image(image2, width = 450)
                        

                    with col3:
                        st.write('')
                    st.markdown('<p style="text-align: center;">Geospatial Information</p>',unsafe_allow_html=True)
                with coll3:
                    col1, col2, col3 = st.columns([0.15,0.8,0.1])
                    with col1:
                        st.write('')

                    with col2:
                        st.image(image3, width = 400)
                        

                    with col3:
                        st.write('')
                    st.markdown('<p style="text-align: center;">Variable of Importance</p>',unsafe_allow_html=True)
        elif choose == "Data Distribution":
            try:
                # create container for input data
                col1, col2 = st.columns([0.8, 0.2])

                # Define format data
                with col2:
                    type_data = st.selectbox('Data Format :',('.parquet','.xlsx','.csv','.geojson'))

                # Read Data
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
                    dataframe = st.session_state['df']
                    if dataframe.shape[0]<100:
                        st.dataframe(dataframe)
                                
                    else:
                        st.dataframe(dataframe.head(10))
                    # get unique values in column
                    column_unique = dataframe.columns.tolist()

                    # get numeric column
                    column_numeric = module_analyze.getNumeric(dataframe)

                    

                    # create selectbox y and x var
                    with st.container():
                        col1, col2, col3 = st.columns([0.1,0.45,0.45])
                        
                        with col2:
                            y1 = st.selectbox('Variable Y1 :', tuple(column_numeric[0]))
                        
                        with col3:
                            y2 = st.selectbox('Variable Y2 :',tuple(column_numeric[0]))
                        
                        with col1:
                            y0 = st.selectbox('Variable X (Unique):', tuple(column_unique))

                    # prep X table
                    df1 = dataframe[[y0]].copy()
                    df1[y0] = df1[y0].astype(str)
                    df1.rename(columns = {y0:'Unique ID Outlet'}, inplace = True)

                    # merging X var with numeric var
                    df_histo = pd.concat([df1, column_numeric[1]], axis = 1)

                    # Histogram 1
                    with st.container():
                        col1, col2 = st.columns([0.85,0.15])
                        with col2:
                            sortby = st.selectbox('Sort By :',tuple([y1]+[y2]))
                            st.title("")
                            st.title("")
                            data_desc = pd.DataFrame({
                                f'Average {y2}':[df_histo[y2].mean()],
                                f'Median {y2}':[df_histo[y2].median()],
                                f'Minimum {y2}':[df_histo[y2].min()],
                                f'Maximum {y2}':[df_histo[y2].max()]
                            }).T
                            data_desc.columns = ['Statistic']
                            st.dataframe(data_desc)
                        with col1:
                            fig1 = visualization.plot_perbandingan(df_histo, 'Unique ID Outlet', y1, y2, sortby)
                            st.plotly_chart(fig1, use_container_width=True)

                    with st.container():
                        col1, col2 = st.columns([0.5, 0.5])
                        with col1:
                            st.text("")
                            st.text("")
                            st.text("")
                            fig = visualization.plot_histogram(df_histo, y1)
                            st.plotly_chart(fig, use_container_width=True, 
                                            sharing="streamlit", theme="streamlit")
                        with col2:
                            st.text("")
                            st.text("")
                            st.text("")
                            fig = visualization.plot_scatter(df_histo, y1,y2)
                            st.plotly_chart(fig, use_container_width=True)
                except:
                    print(404)

            except:
                st.text('No Input Data')

        elif choose == "Correlation Analysis":
            # Branch Selections
            method = st.sidebar.selectbox(
                'Please Select Method',
                ('Bar Chart','Heatmap'), index=0)
            
            try :
                # Title
                st.title('Correlation Analysis') 

                # get dataframe from Histogram process
                dataframe = st.session_state['df']

                # Heatmap Processing
                # get numeric column
                column_numeric = module_analyze.getNumeric(dataframe)[0]
                df_numeric = dataframe[column_numeric]

                # get unique values in column
                column_unique = module_analyze.getObject(dataframe)
                # Bar Chart Processing
                if method == 'Bar Chart':
                    corr_ = st.selectbox('Correlation by :', tuple(column_numeric))
                    correlation_ = module_analyze.getCorrelationdf(df_numeric, corr_)
            
                    col1, col2 = st.columns([0.75,0.25])
                    with col1:
                        fig = visualization.correlation_bar_chart(correlation_, corr_)
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        corr_s = correlation_.sort_values(corr_, ascending = False)[['Keterangan','index',corr_]].round(2)
                        corr_s.rename(columns = {'index':'Parameter',
                                                corr_:'Correlation'
                                                }, inplace = True)
                        corr_s = corr_s[corr_s['Parameter']!=corr_]
                        for i in ['Strong Positive Correlation : > 0.5',
                                    'Moderate Positive Correlation : 0.2 to 0.5',
                                    'Low Positive Correlation : 0.1 to 0.2',
                                    'Low Negative Correlation : -0.2 to -0.1',
                                    'Moderate Negative Correlation : -0.5 to -0.2',
                                    'Strong Negative Correlation : < - 0.5',
                                    'Very low : -0.1 to 0.1']:
                            st.text(i)
                            st.dataframe(corr_s[corr_s['Keterangan']==i][['Parameter','Correlation']])


                elif method == 'Heatmap':
                    # create container
                    col1, col2 = st.columns([0.5,0.5])
                    with col1:
                        # get target col
                        col_target = st.multiselect('Target Column(s)', column_numeric)
                    with col2:
                        # get col input
                        #column_non_target = []
                        #for i in column_numeric:
                        #    if i not in col_target:
                        #        column_non_target.append(i)
                        col_input = st.multiselect("Attribute Column(s):",column_numeric)
                    
                    # get variable from input and target
                    var = []
                    for i in col_input:
                        if i not in col_target:
                            var.append(i)
                    var = var+col_target
                    correlation = dataframe[var].corr()[col_target]
                    correlation = correlation[correlation.index.isin(col_input)].round(2)
                    #if correlation.shape[1]==0 or correlation.shape[1]==0:
                    #    st.text('Fill Attribute and Target Columns')
                    #else:
                    # plot correlation
                    if st.button('Generate'):
                        fig = visualization.heatmap_plot(correlation)
                        fig.update_layout(title = f'Correlation Analysis')
                        st.plotly_chart(fig, use_container_width=True)

            except:
                st.text('No Input Data')

                

        elif choose == "Variable of Importance Analysis":
            # create container for input data
            col1, col2 = st.columns([0.8, 0.2])

            # Define format data
            with col2:
                type_data = st.selectbox('Data Format :',('.parquet','.xlsx','.csv','.geojson'))

            # Read Data
            with col1:
                uploaded_files = st.file_uploader("", accept_multiple_files=True)
                for uploaded_file in uploaded_files:
                    bytes_data = uploaded_file.read()
                    try:
                        dataframe = read_module.read_data(uploaded_file, type_data)
                        st.session_state['df'] = dataframe
                    except:
                        st.text("Can't read format data")

            # Branch Selections
            try:
                # Heatmap Processing
                # get numeric column
                column_numeric = module_analyze.getNumeric(dataframe)[0]
                df_numeric = dataframe[column_numeric]

                # get unique values in column
                column_unique = module_analyze.getObject(dataframe)

                # create container
                try:
                    # get target col
                    col_target = st.selectbox('Target :', tuple(column_numeric))
                    # get col input
                    column_non_target = []
                    for i in column_numeric:
                        if i not in col_target:
                            column_non_target.append(i)
                    col_input = st.multiselect("Attribute Column(s):",column_non_target)

                    # define model
                    model_type = st.selectbox('Model :', ('Linear Regression','Lasso Regression','RandomForest Regression',
                                                    'Decision Tree Regression','XGBoost Regression'
                                                    ))
                    
                    

                    # Processing
                    data = df_numeric[[col_target]+col_input].fillna(0)

                    # define X and Y table
                    X = module_analyze.normalize_target(data, col_input)
                    y = data[col_target]

                    if model_type == 'Linear Regression':
                        model = LinearRegression()
                        model.fit(X, y)
                    
                    elif model_type == 'Lasso Regression':
                        model = linear_model.LassoLars(random_state=42)
                        model.fit(X, y)
                    
                    elif model_type == 'RandomForest Regression':
                        model = RandomForestRegressor(random_state=42)
                        model.fit(X, y)
                    
                    elif model_type == 'Decision Tree Regression':
                        model = DecisionTreeRegressor(random_state=42)
                        model.fit(X, y)
                    
                    elif model_type == 'XGBoost Regression':
                        model = XGBRegressor(random_state=42)
                        model.fit(X, y)
                except:
                    404

                if st.button('Generate VOI'):
                    try:
                        df_eval = pd.DataFrame({
                            'RMSE':[module_analyze.evaluate(model, X,y)[0].round(3)],
                            'R2':[module_analyze.evaluate(model, X,y)[1].round(3)]
                        }).T
                        df_eval.columns = ['error']
                        df_voi = module_analyze.getCoefTable(model, model_type, X)
                        col3, col4 = st.columns([0.3,0.7])
                        with col3:
                            st.dataframe(df_eval)
                            st.dataframe(df_voi[['feature','score']])
                        with col4:
                            fig = visualization.voi_chart(df_voi)
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.text('Failed build VOI')
            except:
                st.text('No Input Data')