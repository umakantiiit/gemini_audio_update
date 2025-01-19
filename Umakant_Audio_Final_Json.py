import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path
import json

# Configure the generative AI API
genai.configure(api_key=st.secrets["gemini_api_key"])

# Helper function to upload files to Gemini
def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Common generation configuration
generation_config = {
    "temperature": 0.3,
    "response_mime_type": "application/json"
}

# Prompts for the models
Prompt_for_audio_transcript = '''
You are an advanced AI assistant specialized in audio processing, speaker diarization, and emotion detection. Your expertise lies in analyzing audio files, identifying speakers, transcribing conversations, and detecting emotions in real-time. Your task is to process an audio file from a call center and provide a detailed, structured output in JSON format.
Task:

Number of Speakers: Identify and state the total number of unique speakers in the audio file.

Transcript with Speaker Labels: Generate a clear and accurate transcript of the audio, labeling each segment of speech with the corresponding speaker (e.g.Agent,Client etc). Use proper punctuation and formatting for readability.

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
        "Speaker": "<Agent/client/Unknown>",
        "Voice": "<extracted_text_from_audio>",
        "Emotion": "<detected_emotion>"
      },
      {
        "Speaker": "<Agent/client/Unknown>",
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
        "Speaker": "Agent",
        "Voice": "Hello, how can I assist you today?",
        "Emotion": "neutral"
      },
      {
        "Speaker": "Client",
        "Voice": "I’m having issues with my recent order.",
        "Emotion": "frustrated"
      },
      {
        "Speaker": "Agent",
        "Voice": "I’m sorry to hear that. Can you provide your order number?",
        "Emotion": "neutral"
      },
      ...
    ]
  }
}
'''

system_prompt_audio = '''You are a highly skilled AI assistant with a deep understanding of audio analysis, natural language processing, and emotional intelligence. You are meticulous, detail-oriented, and committed to delivering accurate and structured results. Your goal is to provide a comprehensive analysis of the call center audio, ensuring the transcript is clear, emotions are accurately detected, and the output is well-organized for further use.'''


system_prompt_json = '''You are an AI trained in analyzing customer service call transcripts. Your expertise lies in emotion detection, summarization, and extracting key insights from conversations. You are meticulous, detail-oriented, and capable of providing structured outputs in JSON format.'''


prompt_transcript_to_output = '''
Analyze the provided JSON input, which contains a customer service call transcript with emotion labels for each speaker. Extract the following details and present them in a structured JSON format:
	1- Emotion Tracking of Clients: A list of emotions expressed by the client (Speaker B) throughout the conversation.
	2-Emotion Tracking of Agents: A list of emotions expressed by the agent (Speaker A) throughout the conversation.
	3-Important Words Used in the Conversation: A list of key words or phrases that are significant to the conversation (e.g., billing, late fee, card expired, etc.).
	4-Questions Asked by the Customer:ANALYSE THIS CAREFULLY.THIS SHOULD INCLUDE WHY A CLIENT CALLED CUSTOMER SERVICE.
	5-Resolutions Given by the Agent: A list of resolutions or actions taken by the agent to address the client's concerns.
 	6-Suggestions For Agents:Analyse carefully what the customer asks and what are the response given by the agent.Then decide what better we can suggest the Agent to improve.
	7-Important Conclusion and Summary of Conversation: A concise summary of the conversation, including the main issue, resolution, and any additional actions taken.
	8-Client Satisfaction: A boolean value (true or false) indicating whether the client seemed satisfied with the agent's response based on their emotions and statements.
	
Input:
The JSON input provided contains the call transcript with speaker labels, their statements, and emotion labels.

Output Format:
Your output must be in JSON format, structured as follows:
{
  "Emotion Tracking of Clients": ["emotion1", "emotion2", ...],
  "Emotion Tracking of Agents": ["emotion1", "emotion2", ...],
  "Important Words Used in the Conversation": ["word1", "word2", ...],
  "Questions Asked by the Customer": ["question1", "question2", ...],
  "Resolutions Given by the Agent": ["resolution1", "resolution2", ...],
  "Suggestion For the Agent" :["Suggestion1", "Suggestion2",...]
  "Important Conclusion and Summary of Conversation": "summary text",
  "Client Satisfaction": true/false
}

Example Output:
Here’s an example of how the output should look:
{
  "Emotion Tracking of Clients": ["confused", "concerned", "neutral", "slightly-regretful", "relieved", "hopeful", "neutral", "grateful"],
  "Emotion Tracking of Agents": ["neutral", "sympathetic", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral"],
  "Important Words Used in the Conversation": ["billing statement", "late fee", "card expired", "waive", "adjustment", "coverage", "savings"],
  "Questions Asked by the Customer": [
    "The client called  because his latest billing statement seems higher than usual, and he don't understand why.",
    "Would you be open to a quick review?"
  ],
  "Resolutions Given by the Agent": [
    "Waived the late fee as a courtesy.",
    "Updated the client's payment method to avoid future issues.",
    "Offered to review the client's current plan for potential savings."
  ],
  "Suggestion For the agent":["Offer multiple option to cut cost of client".....],
  "Important Conclusion and Summary of Conversation": "The client called regarding an unexpectedly high billing statement due to a late fee. The agent identified the issue as a result of an expired card and waived the late fee. The agent also updated the client's payment details and offered to review their current plan for potential savings. The client expressed relief and gratitude.",
  "Client Satisfaction": true
}

Instructions:
	- Carefully analyze the JSON input to extract the required details.
	- Ensure the output is well-structured and adheres to the provided JSON format.
	- Focus on accuracy in emotion tracking, key phrase extraction, and summarization.
	- Use the client's final emotions and statements to determine satisfaction.
'''

# Initialize the model for both tasks
model_audio = genai.GenerativeModel(
    model_name="gemini-1.5-flash-002",
    system_instruction=system_prompt_audio
)

model_json = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    system_instruction=system_prompt_json
)

st.title("Welcome to CurateAI Audio Assistant")

# Placeholder for storing the first API call result
transcript_json = None

# Audio file upload section
uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "aac", "wav", "aiff"], accept_multiple_files=False)

if uploaded_audio is not None:
    file_extension = Path(uploaded_audio.name).suffix.lower()
    valid_extensions = [".mp3", ".aac", ".wav", ".aiff"]
    
    if file_extension not in valid_extensions:
        st.error("AUDIO FILE IS NOT IN VALID FORMAT")
    else:
        # Save the uploaded audio file temporarily
        with open(uploaded_audio.name, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        
        # Upload to Gemini and process
        mime_type = f"audio/{file_extension.strip('.') if file_extension != '.mp3' else 'mpeg'}"
        myaudio = upload_to_gemini(uploaded_audio.name, mime_type=mime_type)
        
        st.audio(uploaded_audio, format=mime_type)
        
        if st.button("View Transcript"):
            response_audio = model_audio.generate_content([myaudio, Prompt_for_audio_transcript], generation_config=generation_config)
            try:
                transcript_json = json.loads(response_audio.text)
                st.json(transcript_json, expanded=True)
                
                st.session_state.transcript_json = transcript_json  # Store the result in session state
                
                # Inform the user the first step is completed
                st.success("GREAT! Transcript generated successfully! You can now proceed to detailed analysis.")
                
            except json.JSONDecodeError:
                st.write("Here is the raw output from the model:")
                st.text(response_audio.text)

# View Detailed Analysis button
if st.session_state.get("transcript_json") is not None:
    if st.button("View Detailed Analysis"):
        transcript_json = st.session_state.transcript_json
        # Convert transcript JSON to string for the second API call
        transcript_str = json.dumps(transcript_json)
        
        # Upload to Gemini and process for detailed analysis
        response_json = model_json.generate_content([transcript_str, prompt_transcript_to_output], generation_config=generation_config)
        try:
            detailed_analysis_json = json.loads(response_json.text)
            st.json(detailed_analysis_json, expanded=True)
            
            # Add download button for final JSON output
            st.download_button(
                label="Download Detailed Analysis JSON",
                data=json.dumps(detailed_analysis_json, indent=4),
                file_name="detailed_analysis.json",
                mime="application/json"
            )
            
        except json.JSONDecodeError:
            st.write("Here is the raw output from the model:")
            st.text(response_json.text)

# Clean up temporary files after session
@st.cache_data()
def get_session_files():
    return []

def remove_temp_files():
    for file_path in get_session_files():
        if os.path.exists(file_path):
            os.remove(file_path)

remove_temp_files()
