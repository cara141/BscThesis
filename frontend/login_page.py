import streamlit as st
import requests

def login_page(backend_url):
    st.title("Login")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Sign In"):
            resp = requests.post(f"{backend_url}/users/login", json={"username": u, "password": p})
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.logged_in = True
                st.session_state.user_id = data["user_id"]
                st.session_state.username = data["username"]
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_u = st.text_input("New Username")
        new_e = st.text_input("Email")
        new_p = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            resp = requests.post(f"{backend_url}/users/register",
                                 json={"username": new_u, "email": new_e, "password": new_p})
            if resp.status_code == 200:
                st.success("Account created! Please login.")
            else:
                st.error(resp.json().get("detail", "Registration failed"))