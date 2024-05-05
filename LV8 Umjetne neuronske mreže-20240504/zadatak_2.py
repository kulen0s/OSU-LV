import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import models

# Učitavanje modela iz datoteke 'model/model.keras'
model = models.load_model("model/model.keras")

# Učitavanje MNIST skupa podataka za testiranje
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizacija podataka na raspon [0, 1] i dodavanje dimenzije za kanale
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

# Predikcije modela na testnom skupu podataka
predictions = model.predict(x_test)
# Pronalaženje klasa s najvećom vjerojatnošću
y_pred_classes = np.argmax(predictions, axis=1)

# Pronalaženje indeksa loše klasificiranih slika
misclassified_indices = np.where(y_pred_classes != y_test)[0]

# Broj slika koje želimo prikazati
num_images_to_display = 5
# Prikaz loše klasificiranih slika
plt.figure(figsize=(12, 8))
for i, idx in enumerate(misclassified_indices[:num_images_to_display]):
    plt.subplot(1, num_images_to_display, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_classes[idx]}")
    plt.axis('off')
plt.show()
