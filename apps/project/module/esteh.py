import streamlit as st
from hydralit import HydraHeadApp

class EstehApp(HydraHeadApp):
    def __init__(self, title='Predict Revenue - Esteh Indonesia', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        # Display content for EstehApp
        st.title(self.title)
        st.write("This is the Esteh Indonesia page.")

        # Add a back button
        if st.button("Back to Projects"):
            # Set session state to navigate back to the project app
            st.session_state.selected_project = None
