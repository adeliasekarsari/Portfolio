import streamlit as st
from streamlit_option_menu import option_menu
from hydralit import HydraHeadApp
from PIL import Image
from apps.modelBuilding.app.modelbuilding import display as modelBuilding
from apps.modelBuilding.app.hyperparameter_tuning import display as HyperparameterTuning

class ModelBuildingApp(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        with st.sidebar:
            choose = option_menu("Sub-Menu", ["Model Automated",
                                                    "HyperparameterTuning"],
                                icons=['book','pin-map-fill','person lines fill','book'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "black"},
                "icon": {"color": "White", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#6BA1D1"},
            }
            
            )
        
        if choose == "Model Automated":
            modelBuilding()
        else:
            HyperparameterTuning()
