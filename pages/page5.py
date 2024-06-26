from rembg import remove
from PIL import Image

input_path = 'sample_data/Capture d’écran 2024-06-26 144534.png'
output_path = 'sample_data/image_sans_arriere_plan.png'

# Ouvrir l'image
input_image = Image.open(input_path)

# Supprimer l'arrière-plan
output_image = remove(input_image)

# Sauvegarder l'image résultante
output_image.save(output_path)