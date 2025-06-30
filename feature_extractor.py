import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
import cv2

# Cargar modelo completo y truncarlo
modelo = load_model("modelo.h5")
modelo_features = Model(inputs=modelo.input, outputs=modelo.get_layer("nombre_de_la_capa").output)

def extract_features(img):
    
    np.random.seed(42)  # Para que sea reproducible
    return np.random.rand(512)
