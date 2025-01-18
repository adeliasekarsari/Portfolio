import streamlit as st
from hydralit import HydraHeadApp
from PIL import Image
from apps.eda_automation.main import EdaApp

class EDAApps(HydraHeadApp):
    def __init__(self, title='Exploratory Data Analysis', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        # Display content for EstehApp
        st.title(self.title)

        # Add a back button
        if st.button("Back to Projects"):
            # Set session state to navigate back to the project app
            st.session_state.selected_project = None

        if st.session_state.get("show_eda_app", False):
            EdaApp().run()  # Call SimilarityApp's run method
            return 

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



        st.title("")
        text = """
        The EDA Wizard is your go-to solution for automated Exploratory Data Analysis. 
        Quickly analyze datasets with built-in Correlation Analysis, Histogram Visualization, 
        and Variable Importance Assessment. Uncover insights effortlessly, visualize data 
        distributions, and identify influential variables. Simplify 
        your data exploration with this intuitive dashboard.
        """
        st.markdown(f"<p class='medium-font'>{text}</p>", unsafe_allow_html=True)

        st.title("")
        st.title("")
        image1 = Image.open(r"./apps/eda_automation/images/histogram.png")
        image2 = Image.open(r"./apps/eda_automation/images/correlation.png")
        image3 = Image.open(r"./apps/eda_automation/images/voi.png")

        # Add image with image desc
        with st.container():
            coll1, coll2, coll3 = st.columns([0.25,0.25,0.25])
            with coll1:
                col1, col2, col3 = st.columns([0.15,0.8,0.1])
                with col1:
                    st.write('')

                with col2:
                    st.image(image1, width = 450)
                    

                with col3:
                    st.write('')
                
                st.markdown('<p style="text-align: center;">Correlation Analysis</p>',unsafe_allow_html=True)
            with coll2:
                col1, col2, col3 = st.columns([0.15,0.8,0.1])
                with col1:
                    st.write('')

                with col2:
                    st.image(image2, width = 450)
                    

                with col3:
                    st.write('')
                st.markdown('<p style="text-align: center;">Geospatial Information</p>',unsafe_allow_html=True)
            with coll3:
                col1, col2, col3 = st.columns([0.15,0.8,0.1])
                with col1:
                    st.write('')

                with col2:
                    st.image(image3, width = 400)
                    

                with col3:
                    st.write('')
                st.markdown('<p style="text-align: center;">Variable of Importance</p>',unsafe_allow_html=True)

        if st.button("Hands-On EDA Apps"):
            st.session_state["show_eda_app"] = True
    
