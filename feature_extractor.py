import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalMaxPooling2D

# Cargar modelo Keras con VGG16 embebido
full_model = load_model("model.h5")
vgg_model = full_model.get_layer("vgg16")
conv_output = vgg_model.get_layer("block5_conv3").output
feature_extractor = Model(inputs=vgg_model.input, outputs=conv_output)

# Función de preprocesamiento de imagen (de .jpg a tensor)
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.stack([img] * 3, axis=-1)  # de (H, W) a (H, W, 3)
    img = img.astype("float32") / 255.0
    return img

# Extraer y reducir features (512,)
def extract_features_from_image(img_tensor):
    fmap = feature_extractor.predict(np.expand_dims(img_tensor, axis=0))  # (1, H, W, C)
    pooled = GlobalMaxPooling2D()(fmap).numpy().squeeze()
    return pooled
