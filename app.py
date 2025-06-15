import json
import time
import tempfile
import numpy as np
import soundfile as sf
import pandas as pd
import streamlit as st
import pyttsx3
import openai
import webrtcvad
import collections

from fuzzywuzzy import fuzz
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils.doc_reader import extract_text
from utils.quiz_generator import generate_quiz_from_doc
from utils.session import init_session, reset_session

st.set_page_config(page_title="AI Quiz Generator", layout="centered")
st.title("📄 AI Quiz Generator with Voice Support")

client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

init_session()

class VADAudioProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(2)
        self.frames = collections.deque(maxlen=20)
        self.sample_rate = 16000
        self.start_time = None

    def recv(self, frame):
        pcm_data = frame.to_ndarray().tobytes()
        is_speech = self.vad.is_speech(pcm_data, self.sample_rate)

        if is_speech:
            if self.start_time is None:
                self.start_time = time.time()
            self.frames.append(pcm_data)
        elif self.frames:
            combined = b"".join(self.frames)
            st.session_state.vad_audio = combined
            st.session_state.vad_triggered = True
            st.session_state.vad_duration = round(time.time() - self.start_time, 2)
            self.frames.clear()
            self.start_time = None
        return frame

if not st.session_state.quiz:
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "txt", "docx"])
    st.session_state.difficulty = st.selectbox("Select difficulty:", ["easy", "medium", "hard"])
    st.session_state.num_questions = st.slider("Number of questions", 1, 50, 10)

    if st.button("Generate Quiz"):
        if not uploaded_file:
            st.warning("⚠️ Please upload a document to generate the quiz.")
        else:
            with st.spinner("Reading document and generating questions..."):
                try:
                    context = extract_text(uploaded_file)
                    if not context:
                        st.error("❌ Could not extract text from this file.")
                    else:
                        raw_output = generate_quiz_from_doc(
                            context,
                            st.session_state.num_questions,
                            st.session_state.difficulty,
                            st.secrets.get("OPENAI_API_KEY")
                        )
                        start = raw_output.find('[')
                        end = raw_output.rfind(']')
                        quiz_json = raw_output[start:end+1]
                        st.session_state.quiz = json.loads(quiz_json)
                        st.session_state.cur = 0
                        st.session_state.score = 0
                        st.session_state.answered_count = 0
                except Exception as e:
                    st.error(f"❌ Error: {e}")

else:
    quiz = st.session_state.quiz
    cur = st.session_state.cur
    total = len(quiz)

    if cur < total:
        q = quiz[cur]
        st.subheader(f"Q{cur+1} of {total}: {q['question']}")

        if st.button("🔊 Read Question Aloud"):
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            tts_text = f"Question {cur+1}: {q['question']}. Options are: "
            for key, val in q["options"].items():
                tts_text += f"{key}. {val}. "
            engine.say(tts_text)
            engine.runAndWait()

        options = list(q["options"].values())
        options.insert(0, "-- Select an answer --")
        selected = st.radio("Choose your answer:", options, key=f"choice_{cur}")

        if st.session_state.get("auto_choice") and selected == "-- Select an answer --":
            selected = st.session_state["auto_choice"]
            st.session_state[f"choice_{cur}"] = selected
            del st.session_state["auto_choice"]
            st.rerun()

        st.markdown("📢 **Voice Instructions:** Speak your answer clearly after clicking Start. No need to click Transcribe.")

        ctx = webrtc_streamer(
            key="vad_stt",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=VADAudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )

        if ctx and ctx.state.playing:
            st.info("🎙 Voice Activity Detection is active... Speak now.")

        if st.session_state.get("vad_triggered"):
            st.session_state.vad_triggered = False
            audio_data = np.frombuffer(st.session_state.vad_audio, dtype=np.int16)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data, 16000)
                with open(f.name, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
            response_text = transcript.text.strip()
            duration = st.session_state.get("vad_duration", 0)
            st.info(f"⏱️ Recording Duration: {duration} seconds")
            st.success(f"🗣 You said: {response_text}")

            if not response_text or len(response_text) < 4 or all(ord(c) > 127 or not c.isalnum() for c in response_text):
                st.warning("⚠️ Transcription unclear or corrupted. Please retry.")
                st.rerun()

            response_clean = response_text.lower()
            matched = None
            max_score = 0

            for key, val in q["options"].items():
                if key.lower() in response_clean or f"option {key.lower()}" in response_clean:
                    matched = val
                    break

            if not matched:
                for val in q["options"].values():
                    score = fuzz.partial_ratio(val.lower(), response_clean)
                    if score > max_score and score > 80:
                        matched = val
                        max_score = score

            if matched:
                st.session_state["auto_choice"] = matched
                st.success(f"✅ Matched to: {matched}")
                st.rerun()
            else:
                st.warning("❗ Could not match speech to any option.")

        col1, col2 = st.columns([1, 1])
        if col1.button("Submit Answer") and not st.session_state.answered:
            if selected == "-- Select an answer --":
                st.warning("⚠️ Please choose an answer before submitting.")
            else:
                correct_option = q["options"][q["correct"]]
                is_correct = selected == correct_option

                st.session_state.selected = selected
                st.session_state.answered = True
                st.session_state.answered_count += 1

                if is_correct:
                    st.success("✅ Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Incorrect. Correct answer: {correct_option}")

                st.info(q["explanation"])
                time.sleep(1.5)
                st.session_state.cur += 1
                st.session_state.answered = False
                st.session_state.selected = None
                st.rerun()

        if col2.button("Complete Quiz"):
            st.session_state.cur = total
            st.rerun()

    else:
        total_possible = 50
        answered = st.session_state.get("answered_count", len(st.session_state.quiz))
        score = st.session_state.score
        st.success(f"🎉 Quiz complete! Score: {score}/{answered} (Normalized: {score}/{total_possible})")

        data = []
        for i, q in enumerate(st.session_state.quiz[:answered]):
            user_choice = st.session_state.get(f"choice_{i}")
            correct_answer = q["options"][q["correct"]]
            data.append({
                "Question": q["question"],
                "User Answer": user_choice,
                "Correct Answer": correct_answer,
                "Correct": "Yes" if user_choice == correct_answer else "No",
                "Explanation": q["explanation"]
            })

        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Quiz Results as CSV",
            data=csv,
            file_name="quiz_results.csv",
            mime="text/csv"
        )

        if st.button("Restart Quiz"):
            reset_session()
