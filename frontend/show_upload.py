import streamlit as st
import requests

def show_upload(backend_url):

    TOP_GENRES = []
    genre_request = requests.get(f"{backend_url}/classifier/classes")
    if genre_request.status_code == 200:
        genre_response = genre_request.json()
        TOP_GENRES = genre_response["classes"]

    st.title("New Classification")
    up = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if up:
        if st.button("Step 1: Extract & Predict"):
            files = {"file": (up.name, up.getvalue(), up.type)}
            r1 = requests.post(f"{backend_url}/classifier/audio", files=files)
            if r1.status_code == 200:
                data = r1.json()
                st.session_state.last_upload = data
                st.success(f"AI suggests: {data['label']}")

        if "last_upload" in st.session_state:
            res = st.session_state.last_upload

            final_genre = st.selectbox("Confirm Genre", TOP_GENRES, index=TOP_GENRES.index(res['label']))

            if st.button("Step 2: Predict Sub-genre & Save"):
                r2 = requests.post(f"{backend_url}/classifier/features",
                                   json={"features": res["features"], "label": final_genre})
                sub_label = r2.json()["label"]

                # save to database
                save_data = {
                    "user_id": st.session_state.user_id,
                    "title": up.name,
                    "main_genre": final_genre,
                    "sub_genre": sub_label,
                    "features": res["features"]
                }
                requests.post(f"{backend_url}/tracks", json=save_data)
                st.success(f"Saved as {sub_label}!")
                del st.session_state.last_upload