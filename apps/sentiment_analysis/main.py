import streamlit as st
from streamlit_option_menu import option_menu
from hydralit import HydraHeadApp

class SentimentApp(HydraHeadApp):

    def __init__(self, title = 'Sentiment Apps', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        st.title('Coming Soon')