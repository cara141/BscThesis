import streamlit as st
import numpy as np
import requests

BACKEND_URL = "http://127.0.0.1:8000"

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
                response = requests.post(BACKEND_URL+"/predict/audio", files=files)
                result = response.json()

                if "error" in result:
                    st.error(result["error"])

                else:
                    st.session_state['features'] = result["features"]
                    st.session_state['label'] = result["label"]

                    st.success(f"Predicted Main Genre: {result['label']}")
            except Exception as e:
                st.error(f"Could not reach model: {str(e)}")

if 'label' in st.session_state:
    st.divider()
    st.subheader(f"Further analyze {uploaded_file.name}")

    if st.button("Predict Sub-Genre"):
        with st.spinner("Extracting features and routing..."):
            payload = {
                "features": st.session_state['features'],
                "label": st.session_state['label']
            }

            try:
                response = requests.post(BACKEND_URL+"/predict/features", json=payload)
                sub_result = response.json()

                if "error" in sub_result:
                    st.error("Something went wrong: " + sub_result["error"])
                else:
                    st.balloons()
                    st.metric(label="Sub-Genre", value=sub_result["label"])

                    with st.expander("View Raw Feature Vector"):
                        st.write(sub_result["features"])


            except Exception as e:
                st.error(f"Prediciton failed: {str(e)}")