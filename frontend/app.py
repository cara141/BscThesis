import streamlit as st
import numpy as np
import requests

BACKEND_URL = "http://127.0.0.1:8000"

available_genres = []

try:
    response = requests.get(BACKEND_URL + "/classifier/classes")
    result = response.json()

    if "error" in result:
        st.error(result["error"])
    else:
        available_genres = result["classes"]
except Exception as e:
    st.warning(f"Could not reach backend: {str(e)}")

st.set_page_config(page_title="Music Genre Classifier")

st.title("Music Genre Classifier")
st.markdown("""
    Upload an audio file to determine its **Main Genre**,
    then further analyze the results to find out a possible **Sub-Genre**.
""")

uploaded_file = st.file_uploader("Choose an audio file (.mp3, .wav, .ogg)", type=["mp3", "wav", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Predict Main Genre"):
        with st.spinner("Extracting features and routing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            try:
                response = requests.post(BACKEND_URL+"/classifier/audio", files=files)
                result = response.json()

                if "error" in result:
                    st.error(result["error"])

                else:
                    result_label = result["label"]
                    
                    st.session_state['features'] = result["features"]
                    st.session_state['label'] = result_label
                    
                    if result_label not in available_genres:
                        available_genres.append(result_label)

                    st.success(f"Predicted Main Genre: {result_label}")
            except Exception as e:
                st.error(f"Could not reach model: {str(e)}")

if 'label' in st.session_state:
    st.divider()

    # Use a columns layout for a cleaner look
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Verify Main Genre")
        # Initialize the dropdown with the index of the predicted label
        try:
            default_index = available_genres.index(st.session_state['label'])
        except ValueError:
            default_index = 0

        # The user can now change the genre manually
        chosen_genre = st.selectbox(
            "If the prediction is incorrect, select the correct genre below:",
            options=available_genres,
            index=default_index
        )

        # Update session state if the user changes the selection
        st.session_state['label'] = chosen_genre

    with col2:
        st.write("###")  # Spacing
        if st.button("Predict Sub-Genre", use_container_width=True):
            with st.spinner(f"Analyzing {st.session_state['label']} specialist..."):
                payload = {
                    "features": st.session_state['features'],
                    "label": st.session_state['label']
                }

                try:
                    response = requests.post(BACKEND_URL + "/classifier/features", json=payload)
                    sub_result = response.json()

                    if "error" in sub_result:
                        st.error("Something went wrong: " + sub_result["error"])
                    else:
                        st.session_state['sub_genre'] = sub_result["label"]
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# Display the final metric if the sub-genre prediction has occurred
if 'sub_genre' in st.session_state:
    st.metric(label=f"Final Sub-Genre ({st.session_state['label']})", value=st.session_state['sub_genre'])
    with st.expander("View Raw Feature Vector"):
        st.write(np.array(st.session_state['features']))