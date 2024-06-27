from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Modèle et tokenizer pour la traduction français vers anglais
fr_to_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
fr_to_en_tokenizer = MarianTokenizer.from_pretrained(fr_to_en_model_name)
fr_to_en_model = MarianMTModel.from_pretrained(fr_to_en_model_name)

# Modèle et tokenizer pour la traduction anglais vers français
en_to_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
en_to_fr_tokenizer = MarianTokenizer.from_pretrained(en_to_fr_model_name)
en_to_fr_model = MarianMTModel.from_pretrained(en_to_fr_model_name)

# Titre de l'application
st.title("Traduction Français à Anglais / Anglais à Français")

# Champ de saisie de texte
user_input = st.text_input("Entrez quelque chose:")

# Choix de la direction de traduction
translation_direction = st.radio("Choisir la direction de traduction :", ("Français vers Anglais", "Anglais vers Français"))

# Bouton pour valider et lancer le processus
if st.button("Valider"):
    if translation_direction == "Français vers Anglais":
        translator = pipeline('translation_fr_to_en', model=fr_to_en_model, tokenizer=fr_to_en_tokenizer, use_auth_token=api_key)
    elif translation_direction == "Anglais vers Français":
        translator = pipeline('translation_en_to_fr', model=en_to_fr_model, tokenizer=en_to_fr_tokenizer, use_auth_token=api_key)

    result = translator(user_input)
    st.write("Résultat de la traduction :")
    st.write(result)








