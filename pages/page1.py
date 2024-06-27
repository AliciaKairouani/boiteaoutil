import transformers
from transformers import pipeline
import soundfile as sf
from moviepy.editor import VideoFileClip
import pydub
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Titre de l'application
st.title("Transcription Audio/Vidéo")

# Chargement du fichier vidéo ou audio
uploaded_file = st.file_uploader("Chargez un fichier vidéo ou audio", type=["mp4", "mp3", "wav"])

if uploaded_file is not None:
    # Sauvegarder le fichier téléchargé
    file_path = f"temp/{uploaded_file.name}"
    audio_path = file_path.replace(".mp4", ".wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Si le fichier est une vidéo, extraire l'audio
    if file_path.endswith(".mp4"):
        try:
            # Charger la vidéo
            video_clip = VideoFileClip(file_path)
            # Extraire l'audio
            audio_clip = video_clip.audio
            # Enregistrer l'audio extrait
            audio_clip.write_audiofile(audio_path)
            # Fermer les clips pour libérer la mémoire
            video_clip.close()
            audio_clip.close()
        except Exception as e:
            st.error(f"Erreur lors de l'extraction audio de la vidéo : {e}")
            st.stop()

    elif file_path.endswith(".mp3"):
        try:
            audio = pydub.AudioSegment.from_mp3(file_path)
            audio.export(audio_path, format='wav')
        except Exception as e:
            st.error(f"Erreur lors de la conversion MP3 vers WAV : {e}")
            st.stop()
    else:
        audio_path = file_path
    
    # Bouton pour lancer la transcription
    if st.button("Transcrire"):
        try:
            transcription_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", use_auth_token=api_key)
            text = transcription_pipe(audio_path)
            text = text.get("text")
            st.write("Transcription :")
            st.write(text)
        except Exception as e:
            st.error(f"Erreur lors de la transcription : {e}")
    
    # Supprimer les fichiers temporaires
    if os.path.exists(file_path):
        os.remove(file_path)
        st.info("Fichier temporaire supprimé : " + file_path)
    if os.path.exists(audio_path) and file_path != audio_path:
        os.remove(audio_path)
        st.info("Fichier temporaire audio supprimé : " + audio_path)
