import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalMaxPooling2D
import pydicom
import os

# Cargar modelo Keras con VGG16 embebido
full_model = load_model("best_model_epoch_19_val_loss_0.3395_val_acc_0.8645.h5")
vgg_model = full_model.get_layer("vgg19")
conv_output = vgg_model.get_layer("block5_conv3").output
feature_extractor = Model(inputs=vgg_model.input, outputs=conv_output)

def preprocess_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".dcm":
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array.astype(np.float32)

        # Normalizar a 0-255
        image -= image.min()
        if image.max() != 0:
            image /= image.max()
        image *= 255.0
        image = image.astype(np.uint8)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (224, 224))
    image = np.stack([image] * 3, axis=-1)  # (H, W) → (H, W, 3)
    image = image.astype("float32") / 255.0
    return image

# Extraer y reducir features (512,)
def extract_features_from_image(img_tensor):
    fmap = feature_extractor.predict(np.expand_dims(img_tensor, axis=0))  # (1, H, W, C)
    pooled = GlobalMaxPooling2D()(fmap).numpy().squeeze()
    return pooled
