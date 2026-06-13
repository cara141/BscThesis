import streamlit as st
import requests


def show_library(backend_url):
    TOP_GENRES = []
    genre_request = requests.get(f"{backend_url}/classifier/classes")
    if genre_request.status_code == 200:
        genre_response = genre_request.json()
        TOP_GENRES = genre_response["classes"]

    st.title("My Music Library")

    # filtering UI
    col1, col2 = st.columns(2)
    with col1:
        search_title = st.text_input("Search by Title")
    with col2:
        search_genre = st.text_input("Filter by Genre")

    # fetch history
    params = {}
    if search_title: params["title"] = search_title
    if search_genre: params["genre"] = search_genre

    resp = requests.get(f"{backend_url}/tracks/{st.session_state.user_id}", params=params)

    if resp.status_code == 200:
        tracks = resp.json()

        # session state will be used to manage track editing
        if "edited_tracks" not in st.session_state:
            st.session_state.edited_tracks = {}

        for track in tracks:
            track_id = track['id']

            current_main = st.session_state.edited_tracks.get(f"main_{track_id}", track['main_genre'])
            current_sub = st.session_state.edited_tracks.get(f"sub_{track_id}", track['sub_genre'])

            with st.expander(f"{track['title']} — {current_main} ({current_sub})"):
                # track edit row
                new_title = st.text_input("Track Title", value=track['title'], key=f"t_{track_id}")

                st.divider()
                st.write("### Change Genre")

                try:
                    current_idx = TOP_GENRES.index(current_main)
                except ValueError:
                    current_idx = 0

                selected_main = st.selectbox(
                    "Change Main Genre",
                    options=TOP_GENRES,
                    index=current_idx,
                    key=f"sel_{track_id}"
                )

                if st.button("Repredict Sub-genre", key=f"re_{track_id}", use_container_width=True):
                    re_payload = {
                        "features": track['features'],
                        "label": selected_main
                    }
                    re_resp = requests.post(f"{backend_url}/classifier/features", json=re_payload)

                    if re_resp.status_code == 200:
                        new_sub = re_resp.json()['label']

                        # change the track genres in the session state
                        st.session_state.edited_tracks[f"main_{track_id}"] = selected_main
                        st.session_state.edited_tracks[f"sub_{track_id}"] = new_sub

                st.divider()

                # save or delete actions
                btn_col1, btn_col2, _ = st.columns([1, 1, 2])

                with btn_col1:
                    if st.button("Save All Changes", key=f"save_{track_id}", type="primary"):
                        # only send genre labels from the session state
                        update_data = {
                            "id": track_id,
                            "user_id": st.session_state.user_id,
                            "title": new_title,
                            "main_genre": current_main,
                            "sub_genre": current_sub,
                            "features": track['features']
                        }

                        put_resp = requests.put(f"{backend_url}/tracks", json=update_data)
                        if put_resp.status_code == 200:
                            st.session_state.edited_tracks.pop(f"main_{track_id}", None)
                            st.session_state.edited_tracks.pop(f"sub_{track_id}", None)
                            st.rerun()
                        else:
                            st.error("Failed to update database record.")

                with btn_col2:
                    if st.button("Delete Track", key=f"del_{track_id}"):
                        requests.delete(f"{backend_url}/tracks/{st.session_state.user_id}/{track_id}")
                        st.session_state.edited_tracks.pop(f"main_{track_id}", None)
                        st.session_state.edited_tracks.pop(f"sub_{track_id}", None)
                        st.rerun()