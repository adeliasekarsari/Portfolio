import streamlit as st
from PIL import Image
import base64
from hydralit import HydraHeadApp

class HomeApp(HydraHeadApp):
    def __init__(self, title="Welcome to Adelia's Portfolio", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        # Function to set background
        def set_background(image_path):
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background: url(data:image/png;base64,{encoded_image});
                    background-size: cover;
                    background-position: center;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

        # Set background image
        set_background("./style/image/home_background.jpg")

        # Main content
        st.markdown(
            """
            <div style='text-align: center; color: white; padding: 60px 0;'>
                <h1 style='font-size: 4em;'>Welcome to My Portfolio</h1>
                <p style='font-size: 1.5em;'>Explore my work, projects, and achievements!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Create columns to center the button
        col1, col2, col3 = st.columns([1.65, 1, 1])  # Use ratio for centering the button

        with col2:
            if st.button("Go to Introduction >>"):
                # Use HydraApp's navigation
                self.do_redirect("Introduction")
