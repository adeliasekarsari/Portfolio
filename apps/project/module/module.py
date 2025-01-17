import streamlit as st
import base64

def text_header1(text):
    return st.markdown(f"""
        <div style="text-align: center; margin-top: 24px;">
            <h1 style="font-size: 2em; color: #ffffff;">{text}</h1>
        </div>
    """, unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode()
