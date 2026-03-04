import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import time

st.set_page_config(layout="wide")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_my_model():
    return load_model("eurosat_model.h5")

model = load_my_model()

classes = ['AnnualCrop','Forest','HerbaceousVegetation','Highway',
           'Industrial','Pasture','PermanentCrop','Residential',
           'River','SeaLake']

# ---------- COLOR MAP ----------
color_map = {
    "Forest": ("#11998e", "#38ef7d"),
    "AnnualCrop": ("#f7971e", "#ffd200"),
    "PermanentCrop": ("#f7971e", "#ffd200"),
    "HerbaceousVegetation": ("#56ab2f", "#a8e063"),
    "Pasture": ("#56ab2f", "#a8e063"),
    "River": ("#36d1dc", "#5b86e5"),
    "SeaLake": ("#36d1dc", "#5b86e5"),
    "Residential": ("#ff8008", "#ffc837"),
    "Industrial": ("#ff8008", "#ffc837"),
    "Highway": ("#8E2DE2", "#4A00E0")
}

# ---------- GLOBAL CSS ----------
st.markdown("""
<style>

.stApp {
background: linear-gradient(to right, #74ebd5, #4facfe);
font-family: 'Segoe UI', sans-serif;
}

.header {
text-align: center;
font-size: 50px;
font-weight: 800;
color: #0b2545;
margin-top: 20px;
}

.sub {
text-align: center;
font-size: 18px;
margin-bottom: 35px;
color: #13315c;
}

.upload-bar {
background: rgba(255,255,255,0.25);
padding: 18px;
border-radius: 40px;
margin-bottom: 15px;
}

.file-chip {
background: rgba(255,255,255,0.7);
padding: 8px 18px;
border-radius: 30px;
display: inline-block;
margin-top: 8px;
font-weight: 500;
}

.img-card img {
border-radius: 18px;
box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

.result-label {
font-size: 45px;
letter-spacing: 2px;
opacity: 0.85;
margin-bottom: 10px;
}

.result-text {
font-size: 48px;
font-weight: 800;
margin-bottom: 12px;
}

.result-box {
padding: 45px;
border-radius: 25px;
text-align: center;
color: white;
}

.conf-badge {
display: inline-block;
padding: 10px 22px;
border-radius: 30px;
font-size: 25px;
background: rgba(255,255,255,0.25);
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="header">🛰 SATELLITE IMAGE CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Deep Learning Based Land Cover Detection</div>', unsafe_allow_html=True)

# ---------- UPLOAD ----------
st.markdown('<div class="upload-bar">', unsafe_allow_html=True)
file = st.file_uploader("Upload Image", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FILE NAME ----------
if file:
    st.markdown(f'<div class="file-chip">📄 {file.name}</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------
if file:

    with st.spinner("Analyzing Image..."):
        time.sleep(1)

    img = load_img(file, target_size=(64,64))
    arr = img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0]
    index = np.argmax(pred)
    conf = pred[index]
    predicted_class = classes[index]

    # dynamic color
    c1, c2 = color_map[predicted_class]

    # dynamic CSS for result card
    st.markdown(f"""
    <style>

    .result-box {{
    background: linear-gradient(135deg, {c1}, {c2});
    padding: 45px;
    border-radius: 25px;
    text-align: center;
    color: white;
    box-shadow: 0 12px 35px {c2};
    }}

    .stProgress > div > div {{
    background: linear-gradient(90deg, {c1}, {c2});
    height: 12px;
    border-radius: 10px;
    }}

    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1,1.2])

    with col1:
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.image(img, width=300)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-box">

        <div class="result-label">PREDICTION</div>

        <div class="result-text">{predicted_class}</div>

        <div class="conf-badge">
        Confidence: {conf*100:.2f}%
        </div>

        </div>
         """, unsafe_allow_html=True)

        # st.progress(float(conf))
print("Created by - Adinath Kalbande")