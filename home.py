import streamlit as st
import os
from dotenv import load_dotenv
from pages import page1, page2, page3, page4, page5

load_dotenv()

st.set_page_config(
    page_title="Boite à outil", page_icon="⚒️", layout="wide"
)

def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

_max_width_()

st.image("logo.png", width=350)

# Dictionnaire de correspondance des pages
PAGES = {
    "🔉 Speach to Text modéle 1": page1,
    "🔊 Speach to Text modéle 2": page2,
    "🗣️ Text to Speach": page3,
    "📖 traduction": page4,
    "📸 background remover": page5,
}

st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Aller à la page", list(PAGES.keys()))

page = PAGES[selection]
page.app()



