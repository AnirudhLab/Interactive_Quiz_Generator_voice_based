import streamlit as st

def init_session():
    if "quiz" not in st.session_state:
        st.session_state.quiz = None
        st.session_state.cur = 0
        st.session_state.score = 0
        st.session_state.answered = False
        st.session_state.selected = None

def reset_session():
    for key in ["quiz", "cur", "score", "answered", "selected"]:
        st.session_state.pop(key, None)
