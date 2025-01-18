import geopandas as gpd
import pandas as pd
import streamlit as st


@st.cache_data
def read_data(dataset, type):
    if type == '.parquet':
        data = pd.read_parquet(dataset)
        return data
    elif type == '.xlsx':
        data = pd.read_excel(dataset)
        return data
    elif type == '.csv':
        data = pd.read_csv(dataset)
        return data
    else:
        st.text("Can't read format data")
    
