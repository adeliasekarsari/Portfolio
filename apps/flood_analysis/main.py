import streamlit as st
from streamlit_option_menu import option_menu
from hydralit import HydraHeadApp

class FloodApp(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        st.title('Coming Soon')