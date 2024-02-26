import os
import streamlit as st
from hydralit import HydraHeadApp
import base64
from PIL import Image

class AboutMe(HydraHeadApp):

    def __init__(self, title = "About Me", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title


    #This one method that must be implemented in order to be used in a Hydralit application.
    #The application must also inherit from the hydrapp class in order to correctly work within Hydralit.
    def run(self):
        col1, col2, col3 = st.columns([0.3,0.55,0.15])
        with col1:

            # CSS styles file
            with open("./style/main.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

            # Profile image file
            with open("style/image/foto_Up.jpg", "rb") as img_file:
                img = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()

            # PDF CV file
            with open("style/Data Analyst 2023 - Adelia Sekarsari.pdf", "rb") as pdf_file:
                pdf_bytes = pdf_file.read()


            # Profile image
            st.write(f"""
            <div class="container">
                <div class="box">
                    <div class="spin-container">
                        <div class="shape">
                            <div class="bd">
                                <img src="{img}" alt="Adelia Sekarsari">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True)
            
            # Subtitle
            st.write(f"""<div class="subtitle" style="text-align: center;">GIS and Data Analytics</div>""", unsafe_allow_html=True)
            
            cola, colb, colc = st.columns([0.3,0.5,0.2])
            with colb:
                # Download CV button
                st.download_button(
                    label="📄 Download my CV",
                    data=pdf_bytes,
                    file_name="Adelia_Sekarsari_CV.pdf",
                    mime="application/pdf",
                )

                st.write("##")


        with col2:
            # About me section
            st.title("")
            st.title("")
            st.write("""
            
            - 📫 How to reach me: adeliasekarsari.as@gmail.com
                    
            - 📳 Contact Person : +623 7423 3600
            
            - 🧑‍💻 I am a graduate engineer from Gadjah Mada University with a strong foundation in statistics and \
                    transformations acquired during my academic tenure. Over the past two years, I have \
                    applied this knowledge as a dedicated data analyst. My expertise spans various domains, including \
                    telecommunications range analysis, retail analytics, model development and deployment, testing \
                    methodologies, and the creation of interactive visualizations using tools such as Plotly, Streamlit\
                    , and Tableau. In pursuit of advancing my skills, I dedicated four months to intensive training as \
                    a data scientist at Rakamin Academy. This experience significantly expanded my understanding of \
                    machine learning techniques and their practical applications in real-world scenarios
            
            """)

            st.write("##")

            
            
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
        st.write(f"""<div class="subtitle" style="text-align: center;">⬅️ Let's Work Together</div>""", unsafe_allow_html=True)