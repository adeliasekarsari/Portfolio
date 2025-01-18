import streamlit as st
from hydralit import HydraHeadApp
from apps.project.module.module import get_base64_image
import pandas as pd
from apps.similarity.main import SimilarityApp

class POIMatchingApp(HydraHeadApp):
    def __init__(self, title='Similarity - POI Matching', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        if st.session_state.get("show_similarity_app", False):
            SimilarityApp().run()  # Call SimilarityApp's run method
            return 
        
        st.title(self.title)

        if st.button("Back to Projects"):
            st.session_state.selected_project = None

        st.markdown("""
        Welcome to the POI Matching Project Explanation! This tool is designed to evaluate the similarity between two sentences, which can represent Points of Interest (POIs) or other textual data. 

        The objective of this dashboard is to streamline the process of identifying whether two textual descriptions represent the same entity. By leveraging advanced similarity algorithms and machine learning, the dashboard helps in automating the matching process, ensuring accuracy and efficiency in data analysis.

        On this page, you can input two sentences, and the system will calculate their similarity using four methods: **Jaccard**, **TF-IDF**, **Levenshtein**, and **Fuzzy**. Based on the calculated scores, the model predicts whether the sentences are "Similar" or "Not Similar."
        """)


        img1 = get_base64_image(f"apps/project/style/poi_matching/poi_main.png")
        st.markdown(
                f"""
                <div style="text-align: center;">
                    <div class="card">
                        <img src="{img1}" style="width: 100%; height: auto;" alt="Project Image">
                    </div>
                </div>
                """,
                
                unsafe_allow_html=True
            )

        if st.button("Hands-On Similarity"):
            st.session_state["show_similarity_app"] = True
    
        st.markdown("""
        ### How It Works
        1. Enter two sentences into the input fields.
        2. The algorithm processes the sentences in the background using four similarity methods:
        - **Jaccard**: Measures overlap between sets of words.
        - **TF-IDF**: Evaluates term importance and similarity.
        - **Levenshtein**: Calculates edit distance between the sentences.
        - **Fuzzy**: Provides a flexible measure of string similarity.

        3. The model, trained on a dataset of sentence pairs, uses these similarity scores as input features to predict whether the sentences are "Similar" or "Not Similar."

        The results include:
        - A similarity status (e.g., "Similar" or "Not Similar").
        - A detailed table showing the similarity score from each method.
        """)


        st.markdown("""
        ### Model Details and Metrics
        The model powering this dashboard is a **Random Forest Classifier** trained on a dataset containing pairs of sentences. Each pair includes similarity scores from Jaccard, TF-IDF, Levenshtein, and Fuzzy methods, along with a target label indicating whether the sentences are similar (1) or not (0).

        #### Model Performance:
        - **Accuracy**: Measures the overall correctness of predictions.
        - **Precision**: Indicates how many of the predicted similar sentences were correct.
        - **Recall**: Captures how many of the actual similar sentences were correctly identified.
        - **F1-Score**: Balances precision and recall into a single metric.

        Below are the performance metrics for the model:
        """)

        metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Train Set": [0.897959, 0.899953, 0.897959, 0.898564],
            "Test Set": [0.895238, 0.901793, 0.895238, 0.896441]
        }

        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)

        # Display Table in Streamlit
        st.table(metrics_df)

