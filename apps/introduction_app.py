import os
import streamlit as st
from hydralit import HydraHeadApp
import base64
from PIL import Image

class IntroductionApp(HydraHeadApp):
    def __init__(self, title="Introduction", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        # Set background color to dark blue
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #171829;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        col2, col1 = st.columns([0.55,0.3])
        with col1:

            # CSS styles file
            with open("./style/main.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

            # Profile image file
            with open("style/image/foto_Up.jpg", "rb") as img_file:
                img = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()

            # PDF CV file
            with open("style//Data Science - Adelia Sekarsari.pdf", "rb") as pdf_file:
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
            st.write(f"""<div class="subtitle" style="text-align: center;">Adelia Sekarsari</div>""", unsafe_allow_html=True)
            
            cola, colb, colc = st.columns([0.3,0.5,0.2])

            with colb:
                # Download CV button
                st.download_button(
                    label="📄 Download my CV",
                    data=pdf_bytes,
                    file_name="Adelia_Sekarsari_CV.pdf",
                    mime="application/pdf",
                )
        with col2:
            # About me section
            st.title("")
            st.title("")
            st.header("HI, WELCOME TO MY PERSONAL PORTFOLIO")
            st.markdown("<hr style='border: 1px solid #ffffff;'>", unsafe_allow_html=True)
            st.write("""I am Data Scientist with 3 years of experience in predictive modeling, machine learning deployment, \
                     and geospatial analysis. A proven track record of delivering business-impacting solutions, including revenue forecasting, \
                     route optimization, and customer analytics. Proficient in building scalable data pipelines, REST APIs, \
                     and AutoML frameworks using Python and Azure Machine Learning. Skilled in creating interactive dashboards with Tableau \
                     and Streamlit, enabling data-driven decision-making. Recognized for integrating geospatial expertise with advanced analytics to \
                     solve complex business challenges and drive measurable growth.
            """)
            # Section separator (Line)
            st.markdown("<hr style='border: 1px solid #ffffff;'>", unsafe_allow_html=True)

            st.header("Professional Experience")

            cola, colb, colc = st.columns([1.95, 1, 0.6])

            with colb:
                # Add image in the center with small size
                st.image("./style/image/HP-Logo-Bhumi-Varta-01.png", width=200)

        cold, cole = st.columns([0.5, 0.5])
        with cold:
            st.write(f"""<div class="subtitle" style="text-align: center;">Data Analyst</div>""", unsafe_allow_html=True)
            st.write("**April 2022 – July 2024**")
            st.markdown("""
            - **Spearheaded** 6 innovative proof-of-concepts (POCs) and delivered 2 impactful projects.
            - **Used SQL and Python** to optimize sales routes, improving performance by 17%.
            - **Collaborated across teams** on 4 projects, including 5 POCs and 20 pre-sales initiatives.
            - **Analyzed mobile data** and integrated cannibalism analysis to improve customer targeting.
            """)

        with cole:
            st.write(f"""<div class="subtitle" style="text-align: center;">Data Scientist</div>""", unsafe_allow_html=True)
            st.write("**August 2024 – Present**")
            st.write("""
            - **Built a revenue prediction model** for 1,089 business locations across Indonesia.
            - **Developed POI matching** analysis using NLP, achieving 92.1% accuracy.
            - **Designed REST APIs** and data pipelines for ML models with Azure Machine Learning.
            - **Created an AutoML framework**, streamlining workflows for Data Analytics.
            - **Delivered real-time analytics** via Azure Functions, ensuring scalability.
            - Collaborated on three major projects, implementing actionable business insights.
            """)


        st.markdown("<hr style='border: 1px solid #ffffff;'>", unsafe_allow_html=True)
        st.header("Course")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.write(f"""<div class="subtitle" style="text-align: center;">2022</div>""", unsafe_allow_html=True)
            col1a, col1b = st.columns([0.45,1])
            with col1b:
                st.image("./style/image/rakamin.webp", width=188)
            st.write(f"""<div class="subtitle" style="text-align: center;">Data Science</div>""", unsafe_allow_html=True)
            st.write(f"""<div class="subtitle" style="text-align: center;">Machine Learning Specialization & Business Intelligence</div>""", unsafe_allow_html=True)

        with col2:
            st.write(f"""<div class="subtitle" style="text-align: center;">2023</div>""", unsafe_allow_html=True)
            col2c, col2b, col2a = st.columns([0.1,0.5,0.8])
            with col2b:
                st.image("./style/image/simplilearn.png", width=150)
                st.image("./style/image/digitalent.png", width=100)
                
            with col2a:
                st.write("")
                st.write(f"""<div class="subtitle" style="text-align: left;">Data Science with Python</div>""", unsafe_allow_html=True)
                st.write("")
                st.write("")
                st.write("")
                st.write(f"""<div class="subtitle" style="text-align: left;">Woman In Tech</div>""", unsafe_allow_html=True)

        
        with col3:
            st.write(f"""<div class="subtitle" style="text-align: center;">2024</div>""", unsafe_allow_html=True)
            col2c, col2b, col2a = st.columns([0.1,0.5,0.8])
            with col2b:
                st.image("./style/image/cisco.png", width=100)
                st.write("")
                st.write("")
                st.image("./style/image/aws.png", width=100)
                
            with col2a:
                st.write(f"""<div class="subtitle" style="text-align: left;">DevNet associate with Cisco</div>""", unsafe_allow_html=True)
                st.write("")
                st.write("")
                st.write(f"""<div class="subtitle" style="text-align: left;">AWS Cloud Developing</div>""", unsafe_allow_html=True)


        st.markdown("<hr style='border: 1px solid #ffffff;'>", unsafe_allow_html=True)

        st.text("")
        st.text("")

        
        st.write(f"""<div class="subtitle" style="text-align: center;">⬅️ Let's Work Together</div>""", unsafe_allow_html=True)
        social_icons = {
            "WhatApps":["https://wa.me/+6282374233600","https://cdn-icons-png.flaticon.com/512/733/733585.png"],
            "LinkedIn": ["https://www.linkedin.com/in/adeliasekarsari/", "https://cdn-icons-png.flaticon.com/512/174/174857.png"],
            "GitHub": ["https://github.com/adeliasekarsari", "https://icon-library.com/images/github-icon-white/github-icon-white-6.jpg"],
            "e-mail":['mailto: adeliasekarsari.as@gmail.com',"https://cdn-icons-png.flaticon.com/512/732/732200.png"]
        }
        social_icons_html = [f"<a href='{social_icons[platform][0]}' target='_blank' style='margin-right: 10px;'><img class='social-icon' src='{social_icons[platform][1]}' width='40' alt='{platform}''></a>" for platform in social_icons]
    
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            {''.join(social_icons_html)}
        </div>""", 
        unsafe_allow_html=True)
