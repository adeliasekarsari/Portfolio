import streamlit as st
from streamlit_option_menu import option_menu
from hydralit import HydraHeadApp

class ProjectApp(HydraHeadApp):

    def __init__(self, title = 'List of Projects', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        st.title('Coming Soon')