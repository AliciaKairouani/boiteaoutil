from transformers import VitsModel, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
import streamlit as st
import scipy.io.wavfile

# Load environment variables
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Load model and tokenizer
try:
    model = VitsModel.from_pretrained("facebook/mms-tts-fra")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fra")
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Synthesis function
def synthesise(text, model):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        return output
    except Exception as e:
        st.error(f"Error during synthesis: {e}")
        return None

# Streamlit app
st.title("Du texte à la parole")

# Text input
user_input = st.text_input("Entrez quelque chose:")

# Validate button
if st.button("Valider"):
    if user_input:  # Check if the user input is not empty
        audio = synthesise(user_input, model)
        if audio is not None:
            output_filename = "user.wav"
            try:
                scipy.io.wavfile.write(output_filename, rate=model.config.sampling_rate, data=audio.numpy())
            except Exception as e:
                st.error(f"Error writing wav file: {e}")
                st.stop()

            try:
                with open(output_filename, "rb") as file:
                    audio_bytes = file.read()

                # Audio playback
                st.audio(audio_bytes, format='audio/wav')

                # Download button
                st.download_button(
                    label="Télécharger le fichier audio",
                    data=audio_bytes,
                    file_name=output_filename,
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Error handling audio file: {e}")
        else:
            st.error("Synthesis failed.")
    else:
        st.warning("Veuillez entrer du texte avant de valider.")
