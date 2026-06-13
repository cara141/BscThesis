import streamlit as st
import requests

from frontend.dashboard import dashboard
from frontend.login_page import login_page

from config import settings

# Config
BACKEND_URL = settings.backend_url

st.set_page_config(page_title="MGC System", layout="wide")


# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None


# Main App Logic
if not st.session_state.logged_in:
    login_page(BACKEND_URL)
else:
    dashboard(BACKEND_URL)