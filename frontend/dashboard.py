import streamlit as st

from frontend.show_library import show_library
from frontend.show_upload import show_upload

def dashboard(backend_url):
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    menu = st.sidebar.radio("Navigation", ["My Library", "Upload & Classify"])

    if menu == "My Library":
        show_library(backend_url)
    else:
        show_upload(backend_url)