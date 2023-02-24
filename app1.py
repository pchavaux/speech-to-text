# import dependencies

# Audio Manipulation
import audioread
import librosa
from pydub import AudioSegment, silence
import youtube_dl
from youtube_dl import DownloadError

# Models
import torch
from transformers import pipeline, HubertForCTC, T5Tokenizer, T5ForConditionalGeneration, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from pyannote.audio import Pipeline

# Others
from datetime import timedelta
import os
import pandas as pd
import pickle
import re
import streamlit as st
import time


def config():
    """
    App Configuration
    This functions sets the page title, its favicon, initialize some global variables (session_state values), displays
    a title, a smaller one, and apply CSS Code to the app.
    """
    # Set config
    st.set_page_config(page_title="Speech to Text", page_icon="📝")

    # Create a Data Directory
    # Will not be executed with AI Deploy because it is indicated in the DockerFile of the app

    if not os.path.exists("../data"):
        os.makedirs("../data")

    # Initialize session state variables
    if 'page_index' not in st.session_state:
        st.session_state['page_index'] = -1  # Handle which page should be displayed (token page, home page, results page, rename page)
        st.session_state['txt_transcript'] = ""  # Save the transcript as .txt so we can display it again on the results page
        st.session_state["process"] = []  # Save the results obtained so we can display them again on the results page
        st.session_state['srt_txt'] = ""  # Save the transcript in a subtitles case to display it on the results page
        st.session_state['srt_token'] = 0  # Is subtitles parameter enabled or not
        st.session_state['audio_file'] = None  # Save the audio file provided by the user so we can display it again on the results page
        st.session_state["start_time"] = 0  # Default audio player starting point (0s)
        st.session_state["summary"] = ""  # Save the summary of the transcript so we can display it on the results page
        st.session_state["number_of_speakers"] = 0  # Save the number of speakers detected in the conversation (diarization)
        st.session_state["chosen_mode"] = 0  # Save the mode chosen by the user (Diarization or not, timestamps or not)
        st.session_state["btn_token_list"] = []  # List of tokens that indicates what options are activated to adapt the display on results page
        st.session_state["my_HF_token"] = "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"  # User's Token that allows the use of the diarization model
        st.session_state["disable"] = False  # Default appearance of the button to change your token

    # Display Text and CSS
    st.title("Speech to Text App 📝")

    st.markdown("""
                    <style>
                    .block-container.css-12oz5g7.egzxvld2{
                        padding: 1%;}
                    # speech-to-text-app > div:nth-child(1) > span:nth-child(2){
                        text-align:center;}
                    .stRadio > label:nth-child(1){
                        font-weight: bold;
                        }
                    .stRadio > div{flex-direction:row;}
                    p, span{ 
                        text-align: justify;
                    }
                    span{ 
                        text-align: center;
                    }
                    """, unsafe_allow_html=True)

    st.subheader("You want to extract text from an audio/video? You are in the right place!")


def load_options(audio_length, dia_pipeline):
    """
    Display options so the user can customize the result (punctuate, summarize the transcript ? trim the audio? ...)
    User can choose his parameters thanks to sliders & checkboxes, both displayed in a st.form so the page doesn't
    reload when interacting with an element (frustrating if it does because user loses fluidity).
    :return: the chosen parameters
    """
    # Create a st.form()
    with st.form("form"):
        st.markdown("""<h6>
            You can transcript a specific part of your audio by setting start and end values below (in seconds). Then, 
            choose your parameters.</h6>""", unsafe_allow_html=True)

        # Possibility to trim / cut the audio on a specific part (=> transcribe less seconds will result in saving time)
        # To perform that, user selects his time intervals thanks to sliders, displayed in 2 different columns
        col1, col2 = st.columns(2)
        with col1:
            start = st.slider("Start value (s)", 0, audio_length, value=0)
        with col2:
            end = st.slider("End value (s)", 0, audio_length, value=audio_length)

        # Create 3 new columns to displayed other options
        col1, col2, col3 = st.columns(3)

        # User selects his preferences with checkboxes
        with col1:
            # Get an automatic punctuation
            punctuation_token = st.checkbox("Punctuate my final text", value=True)

            # Differentiate Speakers
            if dia_pipeline == None:
                st.write("Diarization model unvailable")
                diarization_token = False
            else:
                diarization_token = st.checkbox("Differentiate speakers")

        with col2:
            # Summarize the transcript
            summarize_token = st.checkbox("Generate a summary", value=False)

            # Generate a SRT file instead of a TXT file (shorter timestamps)
            srt_token = st.checkbox("Generate subtitles file", value=False)

        with col3:
            # Display the timestamp of each transcribed part
            timestamps_token = st.checkbox("Show timestamps", value=True)

            # Improve transcript with an other model (better transcript but longer to obtain)
            choose_better_model = st.checkbox("Change STT Model")

        # Srt option requires timestamps so it can matches text with time => Need to correct the following case
        if not timestamps_token and srt_token:
            timestamps_token = True
            st.warning("Srt option requires timestamps. We activated it for you.")

        # Validate choices with a button
        transcript_btn = st.form_submit_button("Transcribe audio!")

    return transcript_btn, start, end, diarization_token, punctuation_token, timestamps_token, srt_token, summarize_token, choose_better_model
access_token="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"
#config()
