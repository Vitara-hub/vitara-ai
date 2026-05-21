import streamlit as st

from modules.nlp import show_nlp_dashboard
from modules.keystroke import show_keystroke_dashboard
from modules.health import show_health_dashboard
# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Vitara Dashboard",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Vitara Dashboard")

dataset = st.sidebar.selectbox(
    "Pilih Dataset",
    [
        "NLP",
        "Keystroke",
        "Health Scoring"
    ]
)

# =========================
# ROUTING
# =========================
if dataset == "NLP":
    show_nlp_dashboard()

elif dataset == "Keystroke":
    show_keystroke_dashboard()

elif dataset == "Health Scoring":
    show_health_dashboard()