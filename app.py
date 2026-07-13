import streamlit as st
import tempfile
import joblib
import pandas as pd
import plotly.express as px

from tensorflow.keras.models import load_model
from utils import extract_features

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

try:
    with open("assets/styles.css") as css:
        st.markdown(
            f"<style>{css.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

@st.cache_resource
def load_resources():

    model = load_model("models/emotion_model.keras")

    encoder = joblib.load("models/label_encoder.pkl")

    return model, encoder


model, label_encoder = load_resources()

emotion_emoji = {

    "Happy": "😄",
    "Sad": "😢",
    "Angry": "😠",
    "Fearful": "😨",
    "Calm": "😌",
    "Neutral": "😐",
    "Disgust": "🤢",
    "Surprised": "😲"

}

st.markdown("""

<div class="fade">

<h1 class="hero-title">

🎙️ Speech Emotion Recognition

</h1>

<p class="hero-subtitle">

Deep Learning • CNN • RAVDESS Dataset

</p>

</div>

""", unsafe_allow_html=True)

left, right = st.columns([1,1])

with left:

    st.markdown("""

<div class="glass">

<h2 class="section-title">

🎵 Upload Audio

</h2>

</div>

""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(

        "Choose WAV File",

        type=["wav"]

    )

    predict = False

    if uploaded_file is not None:

        st.audio(uploaded_file)

        predict = st.button(

            "🎤 Predict Emotion",

            use_container_width=True

        )

with right:

    st.markdown("""

<div class="glass">

<h2 class="section-title">

🧠 Prediction

</h2>

<p style="color:#94A3B8">

Upload an audio file and click the button.

</p>

</div>

""", unsafe_allow_html=True)
   

if predict:

    with st.spinner("🎧 Analyzing emotion..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

            tmp.write(uploaded_file.read())

            audio_path = tmp.name


        mfcc = extract_features(audio_path)

       
        prediction = model.predict(
            mfcc,
            verbose=0
        )

        predicted_index = prediction.argmax()

        predicted_emotion = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = prediction[0][predicted_index] * 100

        emoji = emotion_emoji.get(
            predicted_emotion,
            "🎵"
        )

    emotions = label_encoder.classes_
    scores = prediction[0] * 100

    df = pd.DataFrame({
        "Emotion": emotions,
        "Confidence": scores
    })

    df = df.sort_values(
        by="Confidence",
        ascending=True
    )

    fig = px.bar(
        df,
        x="Confidence",
        y="Emotion",
        orientation="h",
        text="Confidence",
        color="Confidence",
        color_continuous_scale="Viridis"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(

    title=dict(
        text="Emotion Confidence Scores",
        x=0.5,
        font=dict(
            size=24,
            color="white"
        )
    ),

    template="plotly_dark",

    height=450,

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    coloraxis_showscale=False,

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),

    font=dict(
        family="Poppins",
        size=16,
        color="white"
    ),

    xaxis=dict(
        title="Confidence (%)",
        color="white",
        tickfont=dict(size=14)
    ),

    yaxis=dict(
        title="Emotion",
        color="white",
        tickfont=dict(size=14)
    )
)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    
    st.markdown("""
    <div class="glass">

    <h2 class="section-title">
    📋 Prediction Summary
    </h2>

    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        st.success(f"Detected Emotion : **{predicted_emotion}**")

        st.info(f"Confidence : **{confidence:.2f}%**")

    with c2:

        st.info("Dataset : **RAVDESS**")

        st.success("Model : **CNN**")

    st.markdown("<br>", unsafe_allow_html=True)


   
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""

<div class="footer">

<h3 style="color:white">

Speech Emotion Recognition Dashboard

</h3>

<p>

Built with ❤️ using

TensorFlow • Streamlit • Plotly • Librosa

</p>

</div>

""", unsafe_allow_html=True)