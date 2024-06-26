from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

CHECKPOINT = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

def process_input(user_input):
    # Placez ici votre code pour traiter l'entrée utilisateur
    return f"Vous avez entré : {user_input}"

# Titre de l'application
st.title("Traduction Français à Anglais")

# Champ de saisie de texte
user_input = st.text_input("Entrez quelque chose:")

# Bouton pour valider et lancer le processus
if st.button("Valider"):
    translator = pipeline('translation_en_to_fr', model=model.to('cpu'), tokenizer=tokenizer, use_auth_token=api_key)
    result = translator(user_input)
    st.write(result)









