from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from IPython.display import Audio
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", api_key=api_key)
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", api_key=api_key)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", api_key=api_key)

embeddings_dataset = load_dataset("pykeio/librivox-tracks", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"], speaker_embeddings, vocoder=vocoder
    )
    return speech.cpu()


def process_input(user_input):
    # Placez ici votre code pour traiter l'entrée utilisateur
    return f"Vous avez entré : {user_input}"

# Titre de l'application
st.title("Traduction Français à Anglais")

# Champ de saisie de texte
user_input = st.text_input("Entrez quelque chose:")

# Bouton pour valider et lancer le processus
if st.button("Valider"):
    audio = synthesise(user_input)
    Audio(audio, rate=16000)
    st.audio(audio)
