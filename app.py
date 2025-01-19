import apps.modelBuilding
from hydralit import HydraApp
import apps
import streamlit as st

# Set global page config
st.set_page_config(
    page_title="Adelia's Portfolio",
    page_icon="👋🏻",
    layout="wide",
    initial_sidebar_state="auto",
)

if __name__ == "__main__":
    over_theme = {"txc_inactive": "#FFFFFF"}

    # Initialize HydraApp
    app = HydraApp(
        title="Adelia's Portfolio",
        favicon="👋🏻",
        banner_spacing=[5, 30, 60, 30, 5],
        navbar_theme=over_theme,
    )

    # Add apps
    app.add_app("Home", app=apps.HomeApp(title="Home"), is_home=True)
    app.add_app("Projects", app=apps.project.main.ProjectApp(title="Project List"))
    app.add_app("Introduction", app=apps.introduction_app.IntroductionApp(title="Introduction"))
    app.add_app("Exploratory Data Analytics", app=apps.eda_automation.main.EdaApp(title="Exploratory Data Analytics"))
    app.add_app("Similarity Analytics", app=apps.similarity.main.SimilarityApp(title="Similarity Analytics"))
    app.add_app("Sentiment Analysis", app=apps.sentiment_analysis.main.SentimentApp(title="Sentiment Analysis"))
    app.add_app("Machine Learning Automation", app=apps.modelBuilding.main.ModelBuildingApp(title="Machine Learning Automation"))

    # Define complex navigation
    complex_nav = {
        "Home": ["Home"],
        "Introduction": ["Introduction"],
        "Projects": ["Projects"],
        "HandOn": [
            "Similarity Analytics",
            "Exploratory Data Analytics",
            "Machine Learning Automation",
            "Sentiment Analysis"
        ]
    }

    # Run the HydraApp
    app.run(complex_nav)
