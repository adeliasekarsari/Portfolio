import os
import streamlit as st
from hydralit import HydraHeadApp
from PIL import Image

MENU_LAYOUT = [1,1,1,7,2]


class HomeApp(HydraHeadApp):


    def __init__(self, title = "Welcome to Adelia's Portfolio", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title


    #This one method that must be implemented in order to be used in a Hydralit application.
    #The application must also inherit from the hydrapp class in order to correctly work within Hydralit.
    def run(self):
        image1 = Image.open(r".\style\IMAGE\image4.jpg")
        image2 = Image.open(r".\style\IMAGE\image2.jpg")
        image3 = Image.open(r".\style\IMAGE\image3.jpg")
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
        st.markdown("<p class='big-font'>Welcome to </p>", unsafe_allow_html=True)
        st.markdown("<p class='big-font'>Adelia's Dashboard Analytics</p>", unsafe_allow_html=True)

        # Create Space
        st.text("")
        st.text("")

        # Description
        text1 = """Step into a realm of data excellence. As a data analyst,
        this analytics dashboard awaits to elevate your insights. Immerse yourself 
        in unparalleled visualizations, where every data point tells a story, 
        empowering your analytical journey like never before."""

        text2 = """We give the solutions with Data Enrichment, Analysis, and Example of use case. 
        We analyze location with site profiling using Demography, POI, Telco, and SES data."""
        st.markdown(f'<p class="medium-font">{text1}</p>', unsafe_allow_html=True) 
        # st.markdown(f'<p class="medium-font">{text2}</p>', unsafe_allow_html=True)       

        # Create Space
        st.text("")
        st.text("")

        # Add image with image desc
        with st.container():
            coll1, coll2, coll3 = st.columns([0.25,0.25,0.25])
            with coll1:
                col1, col2, col3 = st.columns([0.15,0.8,0.1])
                with col1:
                    st.write('')

                with col2:
                    st.image(image1)
                    

                with col3:
                    st.write('')
                
                st.markdown('<p style="text-align: center;">Disintegrated & disorganised data </p>',unsafe_allow_html=True)
            with coll2:
                col1, col2, col3 = st.columns([0.15,0.8,0.1])
                with col1:
                    st.write('')

                with col2:
                    st.image(image2)
                    

                with col3:
                    st.write('')
                st.markdown('<p style="text-align: center;">Geospatial Information</p>',unsafe_allow_html=True)
            with coll3:
                col1, col2, col3 = st.columns([0.15,0.8,0.1])
                with col1:
                    st.write('')

                with col2:
                    st.image(image3)
                    

                with col3:
                    st.write('')
                st.markdown('<p style="text-align: center;">Insigh and Visualization</p>',unsafe_allow_html=True)

        # Social Icons
        social_icons = {
            # Platform: [URL, Icon]
            "WhatApps":["https://wa.me/+6282374233600","https://cdn-icons-png.flaticon.com/512/733/733585.png"],
            "LinkedIn": ["https://www.linkedin.com/in/adeliasekarsari/", "https://cdn-icons-png.flaticon.com/512/174/174857.png"],
            "GitHub": ["https://github.com/adeliasekarsari", "https://icon-library.com/images/github-icon-white/github-icon-white-6.jpg"],
            "e-mail":['mailto: adeliasekarsari.as@gmail.com',"https://cdn-icons-png.flaticon.com/512/732/732200.png"]
        }

        # Create Space
        st.text("")
        st.text("")

        social_icons_html = [f"<a href='{social_icons[platform][0]}' target='_blank' style='margin-right: 10px;'><img class='social-icon' src='{social_icons[platform][1]}' width='40' alt='{platform}''></a>" for platform in social_icons]
        
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            {''.join(social_icons_html)}
        </div>""", 
        unsafe_allow_html=True)





