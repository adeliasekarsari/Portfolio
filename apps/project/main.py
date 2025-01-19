import streamlit as st
from hydralit import HydraHeadApp
from apps.project.module.module import get_base64_image
from apps.project.module.esteh import EstehApp 
from apps.project.module.poi_matching import POIMatchingApp
from apps.project.module.eda import EDAApps
from apps.business_expansion.main import BEApp
from apps.project.module.automl import AutoMLApp

class ProjectApp(HydraHeadApp):
    def __init__(self, title='List of Projects', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        if "selected_project" not in st.session_state:
            st.session_state.selected_project = None

        if st.session_state.selected_project == "esteh":
            esteh_app = EstehApp()
            esteh_app.run()
        elif st.session_state.selected_project == "poi_matching":
            poi_matching_app = POIMatchingApp()
            poi_matching_app.run()
        elif st.session_state.selected_project == "eda":
            eda_app = EDAApps()
            eda_app.run()
        elif st.session_state.selected_project == 'automl':
            automl_app = AutoMLApp()
            automl_app.run()
        elif st.session_state.selected_project == "huff":
            huff_app = BEApp()
            huff_app.run()
        else:
            # Render the project grid
            self.render_header()
            self.render_projects_grid()

    def render_header(self):
        st.markdown("""
            <div style="text-align: center; margin-top: 50px;">
                <h1 style="font-size: 3em; color: #ffffff;">My Data Science Journey</h1>
                <p style="font-size: 1.2em; color: #ffffff;">
                    A showcase of projects where data meets innovation—turning insights into impactful solutions.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    def render_projects_grid(self):
        """
        Renders a 3x3 grid of project cards with images and headers, removing the 9th card.
        """
        st.markdown(
            """
            <style>
            .card {
                text-align: center;
                background-color: #f9f9f9;
                border-radius: 15px;
                padding: 15px;
                margin: 10px; /* Add margin between cards */
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .card:hover {
                transform: scale(1.05);
                box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.15);
            }
            .card img {
                width: 80%; /* Smaller image */
                height: auto;
                border-radius: 10px;
            }
            .card h3 {
                margin-top: 10px;
                font-size: 0.9em; /* Smaller text */
                color: #333;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        title_dict = {
            "esteh": "Predict Revenue ESTEH Indonesia",
            "poi_matching": "POI Matching - NLP",
            "eda": "Exploratory Data Analysis",
            "automl": "Machine Learning Automation",
            "lippo": "Telco Location - Lippo",
            "huff": "Huff Demand Analysis",
            "demand": "Demand"
        }
        projects = [
            {"title": f"{title_dict[i]}", 
             "image_url": get_base64_image(f"apps/project/style/{i}.png"),
             "project_key": i}
            for i in ['esteh', 'poi_matching', 'eda', 'automl', 'lippo', 'huff', 'demand']
        ]

        projects = projects[:-1]

        for row_index in range(0, len(projects), 3):
            coll, col1, col2, col3, colr = st.columns([0.2, 1, 1, 1, 0.2])

            for col, project in zip([col1, col2, col3], projects[row_index:row_index + 3]):
                with col:
                    self.render_project_card(project)

    def render_project_card(self, project):
        """
        Renders a single project card with an image and a header.
        """
        st.markdown(
            f"""
            <div class="card">
                <img src="{project['image_url']}" alt="{project['title']}" style="width: auto; height: 200px;">
                <h3>{project['title']}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(f"View {project['title']}"):
                st.session_state.selected_project = project['project_key']

if __name__ == "__main__":
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = None
    app = ProjectApp()
    app.run()
