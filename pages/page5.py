from rembg import remove
from PIL import Image
import streamlit as st
import os

# Titre de l'application
st.title("Suppression de l'arrière-plan d'une image")

# Champ de téléchargement de fichier
uploaded_file = st.file_uploader("Chargez une image (PNG ou JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Sauvegarder le fichier téléchargé temporairement
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Ouvrir l'image avec PIL
    input_image = Image.open(file_path)

    # Supprimer l'arrière-plan avec rembg
    output_image = remove(input_image)

    # Afficher l'image résultante dans Streamlit
    st.image(output_image, caption="Image sans arrière-plan", use_column_width=True)

    # Bouton de téléchargement de l'image résultante
    st.markdown("### Télécharger l'image sans arrière-plan")
    st.markdown("Cliquez ci-dessous pour télécharger l'image sans arrière-plan.")
    st.download_button(
        label="Télécharger l'image",
        data=output_image,
        file_name="image_sans_arriere_plan.png",
        mime="image/png"
    )

    # Supprimer le fichier temporaire après utilisation
    if os.path.exists(file_path):
        os.remove(file_path)
