import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path
import json

genai.configure(api_key=st.secrets["gemini_api_key"])

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

generation_config = {
    "temperature": 0.4,
    "response_mime_type": "application/json"
}

Prompt_for_audio_transcript = '''
"You are an advanced AI assistant specialized in audio processing and speaker diarization. Your task is to analyze an audio file and provide the following outputs:

      1-Number of Speakers: Identify and state the total number of unique speakers in the audio file.

      2-Transcript with Speaker Labels: Generate a transcript of the audio, labeling each segment of speech with the corresponding speaker (e.g., Speaker A, Speaker B, Speaker C, etc.). Ensure the transcript is clear, accurate, and easy to read.
      3-For each point in the conversation, note the customer's emotion and provide a timeline in JSON format.
Guidelines:

      -Use clear and concise language.
      -Also Use Proper Punctuations In the Transcript.
      -If there are overlapping speeches, make a note of it and attempt to separate the speakers as accurately as possible.

OUTPUT FORMAT:
      - I need a proper JSON as output
      - The Json structure should represent the complete conversation with Speaker Information alongwith emotion detected at each step.
      {
      Call Details:{
        Number Of Speaker:
        Transcript:{
                 Speaker A:(If u cannot Find out then say Unknown)
                 Voice : Extracted Text From Audio
                 Emotion:
                 Speaker B:(If u cannot Find out then say Unknown)
                 Voice : Extracted Text From Audio
                 Emotion:
                 .........
                    }
      }
      }


'''

model = genai.GenerativeModel(
   model_name="gemini-1.5-pro",
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
            try:
                json_response = json.loads(response.text)
                st.json(json_response,expanded=True)
            except json.JSONDecodeError:
                st.error("Failed to parse JSON response.")

# Clean up temporary files after session
@st.cache_data()
def get_session_files():
    return []

def remove_temp_files():
    for file_path in get_session_files():
        if os.path.exists(file_path):
            os.remove(file_path)

remove_temp_files()
