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
    "temperature": 0.3,
    "response_mime_type": "application/json"
}

Prompt_for_audio_transcript = '''
You are an advanced AI assistant specialized in audio processing, speaker diarization, and emotion detection. Your expertise lies in analyzing audio files, identifying speakers, transcribing conversations, and detecting emotions in real-time. Your task is to process an audio file from a call center and provide a detailed, structured output in JSON format.
Task:

Number of Speakers: Identify and state the total number of unique speakers in the audio file.

Transcript with Speaker Labels: Generate a clear and accurate transcript of the audio, labeling each segment of speech with the corresponding speaker (e.g., Speaker A, Speaker B, Speaker C, etc.). Use proper punctuation and formatting for readability.

Emotion Detection: For each speaker at every point in the conversation, detect and note their emotion (e.g., happy, sad, angry, neutral, frustrated, etc.). Provide a timeline of emotions in JSON format.

Guidelines:
- Use clear, concise, and professional language.
-Ensure the transcript is accurate and easy to read.
-If a speaker cannot be identified, label them as "Unknown."
-Emotions should be detected for each speaker at every conversational turn.
-Follow the JSON output format strictly.

Output Format:
Provide the output in the following JSON structure:
{
  "Call Details": {
    "Number of Speakers": "<total_number_of_speakers>",
    "Transcript": [
      {
        "Speaker": "<Speaker A/Unknown>",
        "Voice": "<extracted_text_from_audio>",
        "Emotion": "<detected_emotion>"
      },
      {
        "Speaker": "<Speaker B/Unknown>",
        "Voice": "<extracted_text_from_audio>",
        "Emotion": "<detected_emotion>"
      },
      ...
    ]
  }
}
Example Output:
{
  "Call Details": {
    "Number of Speakers": 2,
    "Transcript": [
      {
        "Speaker": "Speaker A",
        "Voice": "Hello, how can I assist you today?",
        "Emotion": "neutral"
      },
      {
        "Speaker": "Speaker B",
        "Voice": "I’m having issues with my recent order.",
        "Emotion": "frustrated"
      },
      {
        "Speaker": "Speaker A",
        "Voice": "I’m sorry to hear that. Can you provide your order number?",
        "Emotion": "neutral"
      },
      ...
    ]
  }
}
'''

system_prompt='''You are a highly skilled AI assistant with a deep understanding of audio analysis, natural language processing, and emotional intelligence. You are meticulous, detail-oriented, and committed to delivering accurate and structured results. Your goal is to provide a comprehensive analysis of the call center audio, ensuring the transcript is clear, emotions are accurately detected, and the output is well-organized for further use.'''

model = genai.GenerativeModel(
   model_name="gemini-1.5-pro-002",
   system_instruction=system_prompt
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
                st.json(json_response, expanded=True)
                
                # Add the download button for JSON
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(json_response, indent=4),
                    file_name="transcript.json",
                    mime="application/json"
                )
                
            except json.JSONDecodeError:
                st.write("Here is the raw output from the model:")
                st.text(response.text)

# Clean up temporary files after session
@st.cache_data()
def get_session_files():
    return []

def remove_temp_files():
    for file_path in get_session_files():
        if os.path.exists(file_path):
            os.remove(file_path)

remove_temp_files()
