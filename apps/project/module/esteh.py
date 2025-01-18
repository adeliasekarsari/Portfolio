import streamlit as st
from hydralit import HydraHeadApp
from apps.project.module.module import get_base64_image

class EstehApp(HydraHeadApp):
    def __init__(self, title='Predict Revenue - Esteh Indonesia', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        st.title(self.title)

        if st.button("Back to Projects"):
            st.session_state.selected_project = None

        with st.container():
            col1, col2 = st.columns([0.65,0.35])
            with col1:
                img = get_base64_image(f"apps/project/style/esteh/main.png")
                st.markdown(
                    f"""
                    <div class="card">
                        <img src="{img}" style="width: 95%; height: auto;" alt="Project Image">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown("""

                    In this project, we utilized external monthly revenue data from Esteh Indonesia over the past three months, combined with internal data sources, including:

                    - **POI Density**: residential, financial services, store and equipment, food services, basic formal education, high formal education, non-formal education, etc.
                    - **Minimum Distance** to all POI categories.
                    - **Population Ages**: 15–54 years old.
                    - **SES (Socioeconomic Status)**: high, medium-high, medium-low, and low.
                    - **Telco Data**: collected at a 1,000-meter grid level, specific to the outlet's location.

                    Using data from **1,089 outlets**, we implemented the model at the **grid level**, where each grid represents a catchment area centered at the outlet's centroid (3 km radius). This approach allows us to gather relevant POI, demographic, socioeconomic, and telco data within the catchment area to predict revenue at a granular level.

                    The trained model achieved:
                    - **R²**: 0.882
                    - **RMSE**: 5,668,810

                    With an average monthly revenue of **32,169,108** and a maximum of **96,102,069.5**, this model provides actionable insights. This method is specifically designed to support **business expansion**, enabling data-driven decision-making for future outlet placements and revenue growth strategies.
                    """)
                
        with st.container():
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                griana1 = get_base64_image(f"apps/project/style/esteh/griana_c3.png")
                st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <div class="card">
                                <img src="{griana1}" style="width: 90%; height: auto;" alt="Project Image">
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("""
                This visualization shows an example of grid revenue prediction, highlighting revenue ranges for each grid. Clients can select a city to identify grids with the highest predicted revenue for potential business expansion. This helps in making strategic decisions based on revenue potential across different locations.
                """)
            with col2:
                griana1 = get_base64_image(f"apps/project/style/esteh/griana_c1.png")
                st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <div class="card">
                                <img src="{griana1}" style="width: 52%; height: auto;" alt="Project Image">
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("""
                This chart illustrates the distribution of grids based on predicted revenue categories, such as low, medium, and high revenue. It provides an overview of how grids are grouped, helping clients understand the overall revenue potential in the selected area.
                            """)
            with col3:
                griana1 = get_base64_image(f"apps/project/style/esteh/griana_c2.png")
                st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <div class="card">
                                <img src="{griana1}" style="width: 38%; height: auto;" alt="Project Image">
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("""
                This section highlights the key variables influencing revenue predictions, ranked by their importance. It helps clients understand which factors, such as population density or competitor proximity, have the most significant impact on their business outcomes.
                            """)
                st.markdown("#")
                
        with st.container():
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("### Clustering Analysis")
                cluster1 = get_base64_image(f"apps/project/style/esteh/clustering.png")
                st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <div class="card">
                                <img src="{cluster1}" style="width: 95%; height: auto;" alt="Project Image">
                            </div>
                        </div>
                        """,
                        
                        unsafe_allow_html=True
                    )
            with col2:
                st.markdown("#")
                st.markdown("##")
                st.markdown("""
                            We also conducted a clustering analysis based on Points of Interest (POI), 
                            Socio-Economic Status (SES), demographic data, and telecommunication data. 
                            This analysis helps to identify location patterns based on spatial characteristics. 
                            By leveraging these insights, clients can predict outlet performance and make 
                            informed decisions for business expansion in areas with similar spatial attributes""")
                col3, col4, col5 = st.columns([1,1,1.5])
                with col3:
                    cluster2 = get_base64_image(f"apps/project/style/esteh/clusterinf_c1.png")
                    st.markdown(
                            f"""
                            <div style="text-align: center;">
                                <div class="card">
                                    <img src="{cluster2}" style="width: 100%; height: auto;" alt="Project Image">
                                </div>
                            </div>
                            """,
                            
                            unsafe_allow_html=True
                        )
                with col4:
                    cluster3 = get_base64_image(f"apps/project/style/esteh/clustering_c2.png")
                    st.markdown(
                            f"""
                            <div style="text-align: center;">
                                <div class="card">
                                    <img src="{cluster3}" style="width: 80%; height: auto;" alt="Project Image">
                                </div>
                            </div>
                            """,
                            
                            unsafe_allow_html=True
                        )
                with col5:
                    st.markdown("""
                            The Bar Chart shows the total number of outlets in each cluster, helping to visualize the distribution of outlets across different clusters. 
                                It provides insights into which clusters have more or fewer outlets, highlighting areas with stronger or weaker market presence. 
                                The Spider Chart displays 9 key components that define each cluster, including average population aged 15-54 years, mobile density, 
                                total competitors, and various Points of Interest (POI) such as food services, education, office services, and government services. 
                                It also includes the average percentage of high SES and medium-low SES in each cluster. These attributes allow for a clear comparison 
                                of the different clusters, helping stakeholders understand their defining characteristics and make informed strategic decisions.""")

            

            







    
