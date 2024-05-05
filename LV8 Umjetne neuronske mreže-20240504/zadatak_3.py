import numpy as np
from tensorflow import keras
from PIL import Image

# Učitaj izgrađenu mrežu
model = keras.models.load_model("model/model.keras")

# Učitaj sliku test.png
image_path = 'test.png'
image = Image.open(image_path)

# Prilagodi sliku za mrežu
image = image.convert('L')  # Pretvori sliku u grayscale
image = image.resize((28, 28))  # Prilagodi veličinu slike na 28x28 piksela
image = np.array(image)  # Pretvori sliku u numpy array
image = image.reshape(1, 28, 28, 1)  # Prilagodi obliku ulaznog tenzora

# Klasificiraj sliku pomoću izgrađene mreže
prediction = model.predict(image)

# Ispiši rezultat u terminal
predicted_class = np.argmax(prediction)
print(f"Predviđena klasa: {predicted_class}")
