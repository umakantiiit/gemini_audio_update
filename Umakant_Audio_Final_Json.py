import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path

genai.configure(api_key=st.secrets["gemini_api_key"])

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

generation_config = {
    "temperature": 0.4,
    "response_mime_type": "text/plain"  # Set to text/plain if the output is text
}

Prompt_for_audio_transcript = '''
"You are an advanced AI assistant specialized in audio processing and speaker diarization. Your task is to analyze an audio file and provide the following outputs:

      1-Number of Speakers: Identify and state the total number of unique speakers in the audio file.

      2-Transcript with Speaker Labels: Generate a transcript of the audio, labeling each segment of speech with the corresponding speaker (e.g., Speaker A, Speaker B, Speaker C, etc.). Ensure the transcript is clear, accurate, and easy to read.
      3-For each point in the conversation, note the customer's emotion and provide in text  format provided below.
Guidelines:
      - Donot put all the Speaker A conversation at a time.Transcript should be in Proper format according to timeline.
      -Use clear and concise language.
      -Also Use Proper Punctuations In the Transcript.
      -If there are overlapping speeches, make a note of it and attempt to separate the speakers as accurately as possible.

OUTPUT FORMAT:
        No of Speakers:
     - Speaker A(Name if possible): , Emotion:
    Speaker B(Name if possible):, Emotion :
    ......
      - Provide the analysis in a  readable text format
'''

model = genai.GenerativeModel(
   model_name="gemini-1.5-flash",
)

st.title("Welcome to CurateAI Audio Assistant")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "aac", "wav", "aiff"], accept_multiple_files=False)

if uploaded_file is not None:
    file_extension = Path(uploaded_file.name).suffix.lower()
    valid_extensions = [".mp3", ".aac", ".wav", ".aiff"]
    
    if file_extension not in valid_extensions:
        st.error("AUDIO FILE IS NOT IN VALID FORMAT")
    else:
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Upload to Gemini
        mime_type = f"audio/{file_extension.strip('.') if file_extension != '.mp3' else 'mpeg'}"
        myaudio = upload_to_gemini(uploaded_file.name, mime_type=mime_type)
        
        st.audio(uploaded_file, format=mime_type)
        
        if st.button("View Analysis"):
            response = model.generate_content([myaudio, Prompt_for_audio_transcript], generation_config=generation_config)
            st.text_area("Analysis Result", response.text, height=300)
            
            # Add the download button
            st.download_button(
                label="Download Text Transcript",
                data=response.text,
                file_name="transcript.txt",
                mime="text/plain"
            )

# Clean up temporary files after session
@st.cache_data()
def get_session_files():
    return []

def remove_temp_files():
    for file_path in get_session_files():
        if os.path.exists(file_path):
            os.remove(file_path)

remove_temp_files()
