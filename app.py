import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr
import pickle
import os
import time

# 1. Page Config
st.set_page_config(page_title="AI Moderator", page_icon="🛡️")
st.title("🛡️ Video Content Moderator V2.0")

# 2. Model Loading
@st.cache_resource
def load_assets():
    try:
        with open('models/moderator_model.pkl', 'rb') as f:
            m = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            v = pickle.load(f)
        return m, v
    except:
        return None, None

model, vectorizer = load_assets()

if model is None:
    st.error("Model files not found! Check 'models' folder.")
else:
    st.sidebar.success("✅ System Ready")

# 3. Tabs for switching between Video and Text
tab1, tab2 = st.tabs(["📹 Video Analysis", "✍️ Manual Text Check"])

with tab1:
    uploaded_file = st.file_uploader("Upload Video", type=['mp4'])
    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button('🔍 Start Video Analysis'):
            with st.spinner('Analyzing audio...'):
                ts = str(int(time.time()))
                temp_v, temp_a = f"v_{ts}.mp4", f"a_{ts}.wav"
                with open(temp_v, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    clip = mp.VideoFileClip(temp_v)
                    clip.audio.write_audiofile(temp_a, verbose=False, logger=None)
                    clip.close()
                    
                    r = sr.Recognizer()
                    with sr.AudioFile(temp_a) as source:
                        r.adjust_for_ambient_noise(source)
                        audio = r.record(source)
                    transcript = r.recognize_google(audio, language='en-IN')
                    
                    # Prediction
                    text_vec = vectorizer.transform([transcript.lower()])
                    prediction = model.predict(text_vec)[0]
                    probs = model.predict_proba(text_vec)[0]
                    confidence = float(probs[prediction] * 100)

                    st.divider()
                    labels = {0: "🔴 HATE SPEECH", 1: "🟡 OFFENSIVE", 2: "🟢 SAFE"}
                    color = "red" if prediction == 0 else "orange" if prediction == 1 else "green"
                    st.markdown(f"### Result: :{color}[{labels[prediction]}]")
                    st.info(f"**Transcript:** {transcript}")
                    st.metric("Confidence", f"{confidence:.2f}%")
                except Exception as e:
                    st.error(f"Speech-to-Text failed. Try Manual Tab. Error: {e}")
                
                # Cleanup
                if os.path.exists(temp_v): os.remove(temp_v)
                if os.path.exists(temp_a): os.remove(temp_a)

with tab2:
    st.subheader("Direct Model Testing")
    user_input = st.text_area("Agar video ki transcript galat hai, toh asali text yahan likhein:")
    
    if st.button('⚖️ Check Text Sentiment'):
        if user_input:
            # Direct ML logic
            text_vec = vectorizer.transform([user_input.lower()])
            prediction = model.predict(text_vec)[0]
            probs = model.predict_proba(text_vec)[0]
            confidence = float(probs[prediction] * 100)

            st.divider()
            labels = {0: "🔴 HATE SPEECH", 1: "🟡 OFFENSIVE", 2: "🟢 SAFE"}
            color = "red" if prediction == 0 else "orange" if prediction == 1 else "green"
            
            st.markdown(f"### Verdict: :{color}[{labels[prediction]}]")
            st.metric("Confidence Score", f"{confidence:.2f}%")
            
            if confidence < 60:
                st.warning("Model is unsure. Accuracy might be low.")
        else:
            st.warning("Pehle kuch likhiye toh sahi!")