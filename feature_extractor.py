import numpy as np
import cv2
import pydicom
from datetime import datetime
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalMaxPooling2D
from pydicom.pixel_data_handlers.util import apply_windowing

# --- Intentar cargar Unet real, si no usar recorte básico ---
try:
    from utilidades_cropping_unet import Humerus_model, crop_image

    unet_model = Humerus_model.load_from_checkpoint(
        checkpoint_path="UnetPlusPlus_efficientnet-b2_25.ckpt",
        arch="UnetPlusPlus",
        encoder_name="efficientnet-b2",
        in_channels=3,
        out_classes=1,
        map_location="cpu"
    )
    unet_model.eval()

    def humerus_crop(img):
        return crop_image(img, unet_model)

except ImportError:
    def humerus_crop(img):
        # Recorte centrado básico
        h, w, _ = img.shape
        minor_dim = min(h, w)
        start_h = (h - minor_dim) // 2
        start_w = (w - minor_dim) // 2
        return img[start_h:start_h+minor_dim, start_w:start_w+minor_dim]

# --- Cargar extractor de features con VGG19 ---
full_model = load_model("best_model_epoch_19_val_loss_0.3395_val_acc_0.8645.h5")
vgg_model = full_model.get_layer("vgg19")
conv_output = vgg_model.get_layer("block5_conv3").output
feature_extractor = Model(inputs=vgg_model.input, outputs=conv_output)

# --- Calcular edad desde fecha DICOM ---
def calculate_age_from_dicom(birth_date):
    try:
        birth = datetime.strptime(birth_date, "%Y%m%d")
        today = datetime.today()
        return (today - birth).days // 365
    except:
        return None

# --- Preprocesamiento estilo notebook ---
def dicom_preprocess_like_notebook(dicom, final_size=512, operation='crop', invert=False):
    # 1. Extraer imagen con windowing
    img = apply_windowing(dicom.pixel_array, dicom).astype(np.float32)

    # 2. Invertir intensidades si se solicita
    if invert:
        img = np.max(img) - img

    # 3. Normalizar
    img -= np.min(img)
    img /= np.max(img) if np.max(img) != 0 else 1.0

    # 4. Convertir a RGB
    img_rgb = np.stack([img] * 3, axis=-1)

    # 5. Hacer la imagen cuadrada
    h, w, _ = img_rgb.shape
    if operation == 'crop':
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        img_cropped = img_rgb[start_h:start_h+min_dim, start_w:start_w+min_dim]
    elif operation == 'padding':
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        img_cropped = np.pad(
            img_rgb,
            ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w), (0, 0)),
            mode='constant', constant_values=0
        )
    else:
        raise ValueError("operation must be 'crop' or 'padding'")

    # 6. Redimensionar a tamaño final
    img_resized = cv2.resize(img_cropped, (final_size, final_size)).astype("float32")

    return img_resized

# --- Procesamiento completo: DICOM → RGB listo para VGG ---
def preprocess_dicom_image(dicom, target_size=(300, 300)):
    # Preprocesamiento base como el notebook
    img = dicom_preprocess_like_notebook(
        dicom,
        final_size=512,
        operation='crop',
        invert=False
    )

    # Recorte con modelo Unet o método simple
    img_cropped = humerus_crop(img)

    # Redimensionar para VGG y escalar a [0, 1]
    img_resized = cv2.resize(img_cropped, target_size).astype("float32") / 255.0

    return img_resized

# --- Extraer vector de características ---
def extract_features_from_image(img_tensor):
    fmap = feature_extractor.predict(np.expand_dims(img_tensor, axis=0))
    pooled = GlobalMaxPooling2D()(fmap).numpy().squeeze()
    return pooled
