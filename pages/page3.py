from transformers import VitsModel, AutoTokenizer
import torch
from datasets import load_dataset
import os
from dotenv import load_dotenv
import streamlit as st
import scipy.io.wavfile

load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

model = VitsModel.from_pretrained("facebook/mms-tts-fra")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fra")


def synthesise(text, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    return output


# Titre de l'application
st.title("Du texte à la parole")

# Champ de saisie de texte
user_input = st.text_input("Entrez quelque chose:", key="text_input")

# Bouton pour valider et lancer le processus
if st.button("Valider", key="validate_button"):
    audio = synthesise(user_input, model)
    scipy.io.wavfile.write("user.wav", rate=model.config.sampling_rate, data=audio.numpy())
    with open("user.wav", "rb") as file:
        audio_bytes = file.read()
    
    # Afficher le lecteur audio
    st.audio(audio_bytes, format='audio/wav', key="audio_player")
    
    # Bouton de téléchargement du fichier audio
    st.download_button(
        label="Télécharger le fichier audio",
        data=audio_bytes,
        file_name="user.wav",
        mime="audio/wav",
        key="download_button"
    )
