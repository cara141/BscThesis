import streamlit as st
import requests

# Config
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MGC System", layout="wide")

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None


# --- AUTHENTICATION VIEW ---
def login_page():
    st.title("Login")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Sign In"):
            resp = requests.post(f"{BACKEND_URL}/users/login", json={"username": u, "password": p})
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
            resp = requests.post(f"{BACKEND_URL}/users/register",
                                 json={"username": new_u, "email": new_e, "password": new_p})
            if resp.status_code == 200:
                st.success("Account created! Please login.")
            else:
                st.error(resp.json().get("detail", "Registration failed"))


# --- DASHBOARD VIEW ---
def dashboard():
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    menu = st.sidebar.radio("Navigation", ["My Library", "Upload & Classify"])

    if menu == "My Library":
        show_library()
    else:
        show_upload()

def show_library():

    TOP_GENRES = []
    genre_request = requests.get(f"{BACKEND_URL}/classifier/classes")
    if genre_request.status_code == 200:
        genre_response = genre_request.json()
        TOP_GENRES = genre_response["classes"]

    st.title("📚 My Music Library")

    # Filtering UI
    col1, col2 = st.columns(2)
    with col1:
        search_title = st.text_input("Search by Title")
    with col2:
        search_genre = st.text_input("Filter by Genre")

    # Fetch History
    params = {}
    if search_title: params["title"] = search_title
    if search_genre: params["genre"] = search_genre

    resp = requests.get(f"{BACKEND_URL}/tracks/{st.session_state.user_id}", params=params)

    if resp.status_code == 200:
        tracks = resp.json()
        for track in tracks:
            with st.expander(f"{track['title']} — {track['main_genre']} ({track['sub_genre']})"):
                # Track Metadata Edit Row
                new_title = st.text_input("Track Title", value=track['title'], key=f"t_{track['id']}")

                st.divider()
                st.write("### Change Genre")

                # Find the current index of the track's genre in our list
                try:
                    current_idx = TOP_GENRES.index(track['main_genre'])
                except ValueError:
                    current_idx = 0

                selected_main = st.selectbox(
                    "Change Main Genre",
                    options=TOP_GENRES,
                    index=current_idx,
                    key=f"sel_{track['id']}"
                )

                if st.button("Repredict Sub-genre", key=f"re_{track['id']}", use_container_width=True):
                    # Use the NEWLY selected_main from the selectbox
                    re_payload = {
                        "features": track['features'],
                        "label": selected_main
                    }
                    re_resp = requests.post(f"{BACKEND_URL}/classifier/features", json=re_payload)

                    if re_resp.status_code == 200:
                        new_sub = re_resp.json()['label']
                        # Update local loop variable so 'Save Changes' sees it
                        track['main_genre'] = selected_main
                        track['sub_genre'] = new_sub
                        st.success(f"New Sub-genre: {new_sub}")

                st.divider()

                # Final Actions Row
                btn_col1, btn_col2, _ = st.columns([1, 1, 2])

                with btn_col1:
                    if st.button("Save All Changes", key=f"save_{track['id']}", type="primary"):
                        update_data = {
                            "id": track['id'],
                            "user_id": st.session_state.user_id,
                            "title": new_title,
                            "main_genre": track['main_genre'],  # Takes either original or repredicted
                            "sub_genre": track['sub_genre'],
                            "features": track['features']
                        }
                        requests.put(f"{BACKEND_URL}/tracks", json=update_data)
                        st.toast("Record updated successfully!")
                        st.rerun()

                with btn_col2:
                    if st.button("🗑️ Delete Track", key=f"del_{track['id']}"):
                        requests.delete(f"{BACKEND_URL}/tracks/{st.session_state.user_id}/{track['id']}")
                        st.rerun()


def show_upload():

    TOP_GENRES = []
    genre_request = requests.get(f"{BACKEND_URL}/classifier/classes")
    if genre_request.status_code == 200:
        genre_response = genre_request.json()
        TOP_GENRES = genre_response["classes"]

    st.title("New Classification")
    up = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if up:
        if st.button("Step 1: Extract & Predict"):
            files = {"file": (up.name, up.getvalue(), up.type)}
            r1 = requests.post(f"{BACKEND_URL}/classifier/audio", files=files)
            if r1.status_code == 200:
                data = r1.json()
                st.session_state.last_upload = data
                st.success(f"AI suggests: {data['label']}")

        if "last_upload" in st.session_state:
            res = st.session_state.last_upload

            final_genre = st.selectbox("Confirm Genre", TOP_GENRES, index=TOP_GENRES.index(res['label']))

            if st.button("Step 2: Predict Sub-genre & Save"):
                r2 = requests.post(f"{BACKEND_URL}/classifier/features",
                                   json={"features": res["features"], "label": final_genre})
                sub_label = r2.json()["label"]

                # Save to DB
                save_data = {
                    "user_id": st.session_state.user_id,
                    "title": up.name,
                    "main_genre": final_genre,
                    "sub_genre": sub_label,
                    "features": res["features"]
                }
                requests.post(f"{BACKEND_URL}/tracks", json=save_data)
                st.success(f"Saved as {sub_label}!")
                del st.session_state.last_upload


# Main App Logic
if not st.session_state.logged_in:
    login_page()
else:
    dashboard()