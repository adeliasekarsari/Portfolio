import streamlit as st
from streamlit_option_menu import option_menu
from hydralit import HydraHeadApp
from apps.similarity.module import preprocess_text, combined_similarity

class SimilarityApp(HydraHeadApp):

    def __init__(self, title='Similarity Apps', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        st.title('Similarity Analysis')

        text = """
        In this project, I developed a similarity analysis model to identify
        and quantify the degree of similarity between two sentences, leveraging 
        natural language processing techniques to compare semantic meaning and syntactic
        structure, providing valuable insights for applications such
        as content matching, plagiarism detection, and recommendation systems.
        """

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

        st.markdown(f'<p class="medium-font">{text}</p>', unsafe_allow_html=True) 
        st.text("")
        st.text("")

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown(f'<p class="medium-font">Sentence 1</p>', unsafe_allow_html=True)
            s1 = st.text_area("Enter your text below:", height=200)

        with col2:
            st.markdown(f'<p class="medium-font">Sentence 2</p>', unsafe_allow_html=True)
            s2 = st.text_area("Enter your text below: ", height=200)

        # Check if both sentences are provided
        if s1 and s2:
            if st.button('Generate Similarity'):
                # Preprocess the sentences
                processed_s1 = preprocess_text(s1)
                processed_s2 = preprocess_text(s2)

                # Calculate similarity
                similarity_score = combined_similarity(processed_s1, processed_s2)

                # Display the result
                st.success(f'The similarity score between the two sentences is: {similarity_score:.2f}')
        else:
            st.warning('Please enter text in both Sentence 1 and Sentence 2.')