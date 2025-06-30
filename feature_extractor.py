import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
import cv2

# Cargar modelo completo y truncarlo
modelo = load_model("modelo.h5")
modelo_features = Model(inputs=modelo.input, outputs=modelo.get_layer("nombre_de_la_capa").output)

def extract_features(img):
    # Suponiendo img en formato OpenCV (grayscale)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    img = np.repeat(img, 3, axis=-1)    # (H, W, 3)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # (1, H, W, C)
    
    features = modelo_features.predict(img)
    return features.squeeze()
